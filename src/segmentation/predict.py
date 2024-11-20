import os
import cv2
import json
import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from tqdm import tqdm
import numpy as np

class CellSegmentationInference:
    def __init__(self, model_path, cell_type):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Lower threshold for testing
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.INPUT.FORMAT = "BGR"
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = DefaultPredictor(self.cfg)
        print(f"Loaded model from {model_path}")

    def process_video(self, video_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # Get all jpg files
        image_files = sorted([f for f in os.listdir(video_dir) if f.endswith('_ch00.jpg')])
        print(f"Found {len(image_files)} images in {video_dir}")
        
        for img_name in tqdm(image_files):
            img_path = os.path.join(video_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed to load {img_path}")
                continue
            
            # Debug image size
            print(f"Processing image {img_name} with shape {img.shape}")
            
            # Run inference
            outputs = self.predictor(img)
            
            # Extract predictions
            instances = outputs["instances"].to("cpu")
            print(f"Found {len(instances)} instances in {img_name}")
            
            if len(instances) > 0:
                masks = instances.pred_masks.numpy()
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.numpy()
                
                frame_annotations = []
                for i in range(len(masks)):
                    contours, _ = cv2.findContours(
                        masks[i].astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    if contours:
                        contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(contour)
                        
                        if area > 50:  # Reduced area threshold
                            segmentation = contour.flatten().tolist()
                            frame_annotations.append({
                                "bbox": boxes[i].tolist(),
                                "segmentation": [segmentation],
                                "score": float(scores[i]),
                                "category": 1
                            })
                
                if frame_annotations:
                    results[img_name] = frame_annotations
                    print(f"Saved {len(frame_annotations)} annotations for {img_name}")
            
        # Save results
        output_path = os.path.join(output_dir, 'predictions.json')
        with open(output_path, 'w') as f:
            json.dump(results, f)
        print(f"Saved predictions to {output_path}")
        
        return results

def main():
    base_dir = "/scratch/rl4789/CellClassif"
    # cell_types = ["MDA", "FB"]
    cell_types = ["M1", "M2"]
    
    for cell_type in cell_types:
        print(f"\nProcessing {cell_type} videos...")
        model_path = f"{base_dir}/models/{cell_type}/model_final.pth"
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        
        predictor = CellSegmentationInference(model_path, cell_type)
        raw_dir = os.path.join(base_dir, "data/raw", cell_type)
        
        for video_id in os.listdir(raw_dir):
            video_path = os.path.join(raw_dir, video_id)
            if not os.path.isdir(video_path):
                continue
                
            print(f"\nProcessing video {video_id}")
            output_dir = os.path.join(
                base_dir, 
                "data/interim/segmentation",
                cell_type,
                "predictions",
                video_id
            )
            
            predictor.process_video(video_path, output_dir)

if __name__ == "__main__":
    main()