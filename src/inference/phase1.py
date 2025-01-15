# phase1.py

import torch
import json
import os
import cv2
import gc
from typing import Dict, List
from tqdm import tqdm
import numpy as np
from models import MotionFeatureNet  # Import the model from models.py
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from deep_sort_realtime.deepsort_tracker import DeepSort

class CustomPredictor:
    """Custom predictor class with more controlled initialization"""
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.input_format = cfg.INPUT.FORMAT
        
    def __call__(self, original_image):
        """
        Args:
            original_image: Either a numpy array of shape (H, W, C) or a dict with preprocessed inputs
        Returns:
            predictions (dict): the output of the model
        """
        with torch.no_grad():
            if isinstance(original_image, dict):
                # Already preprocessed input
                inputs = original_image
            else:
                # Convert image to the format expected by the model
                if self.input_format == "RGB":
                    original_image = original_image[:, :, ::-1]
                
                height, width = original_image.shape[:2]
                image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
                inputs = {"image": image, "height": height, "width": width}
            
            predictions = self.model([inputs])[0]
            return predictions
        

class Phase1_FrameProcessor:
    """
    Phase 1: Frame Processing
    - Processes each frame to perform detection and tracking.
    - Saves segmentation results and tracking data as JSON files.
    """

    def __init__(
        self, 
        mixed_detectron_path: str,
        output_dir: str,
        cell_type: str,
        video_id: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.output_dir = output_dir
        self.cell_type = cell_type
        self.video_id = video_id
        self.tracking_data = {}  # To accumulate tracking data
        os.makedirs(output_dir, exist_ok=True)

        # Define class mapping (adjust according to your model's classes)
        self.class_mapping = {
            0: "MDA",
            1: "FB",
            2: "M1",
            3: "M2"
        }

        # Initialize Detectron2
        print("=== Phase 1: Initializing Detectron2 ===")
        self.detectron_cfg = get_cfg()
        self.detectron_cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
        self.detectron_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # Adjust based on your dataset
        self.detectron_cfg.MODEL.DEVICE = self.device
        self.detectron_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Detection threshold
        self.detectron_cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
        self.detectron_cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500
        self.detectron_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        self.detectron_cfg.INPUT.MIN_SIZE_TEST = 800
        self.detectron_cfg.INPUT.MAX_SIZE_TEST = 1333

        print("Building Detectron2 model...")
        try:
            # Build model explicitly
            model = build_model(self.detectron_cfg)
            model.eval()  # Set to evaluation mode

            print("Loading Detectron2 model weights...")
            checkpointer = DetectionCheckpointer(model)
            checkpointer.load(mixed_detectron_path)  # Load weights only

            # Move model to device
            model.to(self.device)

            # Create predictor
            self.detectron_predictor = CustomPredictor(self.detectron_cfg, model)
            print("Detectron2 model loaded successfully.")
        except Exception as e:
            print(f"Error during Detectron2 initialization: {e}")
            raise

        # Initialize DeepSort tracker
        print("Initializing DeepSort tracker...")
        self.tracker = DeepSort(
            max_age=30, 
            n_init=3, 
            nms_max_overlap=1.0, 
            embedder='mobilenet', 
            half=True
        )
        print("DeepSort tracker initialized.")

    def process_video_frames(self, image_dir: str, inference_folder: str):
        """
        Processes all frames in the specified directory.
        Saves segmentation results and tracking data as JSON files.
        """
        frame_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
        print(f"Found {len(frame_files)} frames in {image_dir}")

        # Initialize JSON files
        segmentation_output_file = os.path.join(inference_folder, "segmentation_results.json")
        tracking_output_file = os.path.join(inference_folder, "tracking_data.json")

        # Create or clear JSON files
        with open(segmentation_output_file, 'w') as f:
            pass  # Empty the file
        with open(tracking_output_file, 'w') as f:
            pass  # Empty the file

        # Process each frame
        for frame_idx, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
            frame_path = os.path.join(image_dir, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"Warning: Could not read frame {frame_file}. Skipping.")
                continue

            frame_results = self.process_single_frame(frame, frame_idx, inference_folder)

            # Periodic memory management
            if frame_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            # Debug memory usage
            if self.device == 'cuda' and frame_idx % 5 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"Frame {frame_idx} GPU memory: {allocated:.2f}GB")

        # After processing all frames, save tracking data
        with open(tracking_output_file, 'w') as f:
            json.dump(self.tracking_data, f, indent=4)
        print(f"Tracking data saved to {tracking_output_file}")

    def process_single_frame(self, frame: np.ndarray, frame_idx: int, inference_folder: str) -> List[Dict]:
        """
        Processes a single frame: detection and tracking.
        Saves detection results to segmentation_output_file.
        Updates tracking_data.
        """
        print(f"\nProcessing frame {frame_idx}")
        if self.device == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"Initial GPU memory: {allocated:.2f}GB")

        # Resize frame if too large
        max_size = 1333
        height, width = frame.shape[:2]
        scale = min(max_size / max(height, width), 1.0)
        if scale < 1.0:
            new_height = int(height * scale)
            new_width = int(width * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            print(f"Resized frame from {height}x{width} to {new_height}x{new_width}")
        else:
            new_height, new_width = height, width

        with torch.no_grad():
            # Perform detection
            detectron_outputs = self.detectron_predictor(frame)

            if self.device == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"After detection GPU memory: {allocated:.2f}GB")

            # Extract instances
            instances = detectron_outputs["instances"].to("cpu")
            del detectron_outputs
            torch.cuda.empty_cache()

            detections = []
            frame_results = []

            if len(instances) > 0:
                # Process each detected instance
                for i in range(len(instances)):
                    box = instances.pred_boxes.tensor[i].tolist()  # [x1, y1, x2, y2]
                    score = instances.scores[i].item()
                    class_id = instances.pred_classes[i].item()
                    mask = instances.pred_masks[i].cpu().numpy()

                    # Process mask to get contours
                    contours, _ = cv2.findContours(
                        (mask * 255).astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        segmentation = largest_contour.flatten().tolist()

                        detection = {
                            "bbox": box,
                            "score": score,
                            "class_id": class_id,
                            "segmentation": segmentation
                        }
                        detections.append(detection)
                        frame_results.append(detection)

            # Debugging: Print number of detections
            print(f"Number of detections in frame {frame_idx}: {len(detections)}")

            if len(detections) == 0:
                print(f"No detections found in frame {frame_idx}. Skipping tracking.")
            else:
                # Transform detections for DeepSort
                transformed_detections = []
                for det in detections:
                    bbox = det['bbox']
                    score = det['score']
                    class_id = det['class_id']
                    class_label = self.class_mapping.get(class_id, "unknown")  # Default to "unknown"

                    # Ensure that bbox has exactly 4 coordinates
                    if len(bbox) == 4:
                        transformed_detections.append((
                            [bbox[0], bbox[1], bbox[2], bbox[3]],  # Bounding box
                            score,                                   # Detection score
                            class_label                              # Class label
                        ))
                    else:
                        print(f"Invalid bbox format for detection: {bbox}")

                # Debugging: Print the type and structure of transformed_detections
                print(f"Type of transformed_detections: {type(transformed_detections)}")
                if len(transformed_detections) > 0:
                    print(f"Type of first detection: {type(transformed_detections[0])}")
                    print(f"Length of first detection: {len(transformed_detections[0])}")
                    print(f"First detection contents: {transformed_detections[0]}")

                # Update tracker
                tracks = self.tracker.update_tracks(transformed_detections, frame=frame)

                # Assign track IDs and accumulate tracking data
                for track, result in zip(tracks, frame_results):
                    if track.is_confirmed():
                        track_id = str(int(track.track_id))  # Convert to string for JSON compatibility
                        result["track_id"] = track_id

                        if track_id not in self.tracking_data:
                            self.tracking_data[track_id] = []

                        # Extract center coordinates from bbox
                        bbox = result['bbox']
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2

                        self.tracking_data[track_id].append({
                            'frame': frame_idx,
                            'x': center_x,
                            'y': center_y,
                            'bbox': result['bbox'],
                            'segmentation': result['segmentation'],
                            'score': result['score'],
                            'class_id': result['class_id']
                        })
                    else:
                        result["track_id"] = -1  # Unconfirmed tracks

            # Save segmentation results to JSON
            segmentation_output_file = os.path.join(inference_folder, "segmentation_results.json")
            with open(segmentation_output_file, 'a') as f:
                json.dump({
                    "frame_idx": frame_idx,
                    "detections": frame_results
                }, f)
                f.write('\n')

            return frame_results
