# perclass_eval.py

import numpy as np
import torch
import os
import json
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from tqdm import tqdm
from collections import defaultdict

# Configuration
BASE_DIR = "/scratch/rl4789/CellClassif"
CELL_TYPES = ["MDA", "FB", "M1", "M2"]  # All cell types

def load_specialized_predictions(cell_type):
    """Load predictions from specialized Detectron models with improved filtering"""
    predictions_base = os.path.join(BASE_DIR, "data", "interim", "segmentation", cell_type, "predictions")
    raw_images_base = os.path.join(BASE_DIR, "data", "raw", cell_type)
    dataset_dicts = []
    
    # Confidence threshold for including predictions
    CONFIDENCE_THRESHOLD = 0.8
    
    for video_dir in tqdm(os.listdir(predictions_base), desc=f"Loading {cell_type} predictions"):
        pred_file = os.path.join(predictions_base, video_dir, "predictions.json")
        if not os.path.exists(pred_file):
            continue
            
        with open(pred_file, 'r') as f:
            video_predictions = json.load(f)
            
        for image_name, predictions in video_predictions.items():
            image_path = os.path.join(raw_images_base, video_dir, image_name)
            if not os.path.exists(image_path):
                continue
                
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            height, width = img.shape[:2]
            
            record = {
                "file_name": image_path,
                "image_id": len(dataset_dicts),
                "height": height,
                "width": width,
                "annotations": []
            }
            
            valid_annotations = []
            for pred in predictions:
                # Filter by confidence score
                if pred["score"] < CONFIDENCE_THRESHOLD:
                    continue
                    
                # Verify bbox format and contents
                bbox = pred["bbox"]
                if len(bbox) != 4:
                    continue
                    
                x1, y1, x2, y2 = bbox
                # Convert to XYWH format
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                # Skip invalid boxes
                if bbox_width <= 0 or bbox_height <= 0:
                    continue
                    
                # Verify segmentation
                if not pred["segmentation"] or not isinstance(pred["segmentation"][0], list):
                    continue
                
                # Ensure segmentation points are within image bounds
                seg_points = np.array(pred["segmentation"][0]).reshape(-1, 2)
                if np.any(seg_points < 0) or np.any(seg_points[:, 0] >= width) or np.any(seg_points[:, 1] >= height):
                    continue
                
                annotation = {
                    "bbox": [x1, y1, bbox_width, bbox_height],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": pred["segmentation"],
                    "category_id": CELL_TYPES.index(cell_type),
                    "iscrowd": 0,
                    "area": bbox_width * bbox_height
                }
                valid_annotations.append(annotation)
            
            if valid_annotations:  # Only add images with valid predictions
                record["annotations"] = valid_annotations
                dataset_dicts.append(record)
    
    print(f"Loaded {len(dataset_dicts)} valid images for {cell_type}")
    return dataset_dicts

def register_datasets():
    """Register datasets from specialized predictions"""
    # Clear existing registrations to avoid duplication
    for d in ["cell_train", "cell_val"]:
        if d in DatasetCatalog:
            del DatasetCatalog._REGISTERED[d]
            del MetadataCatalog._REGISTERED[d]
    
    all_train_data = []
    all_val_data = []
    
    for cell_type in CELL_TYPES:
        print(f"\nProcessing {cell_type} dataset...")
        dataset_dicts = load_specialized_predictions(cell_type)
        
        if not dataset_dicts:
            print(f"Warning: No valid data found for {cell_type}")
            continue
        
        # Split into train/val (80/20)
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(dataset_dicts)
        split_idx = int(len(dataset_dicts) * 0.8)
        train_data = dataset_dicts[:split_idx]
        val_data = dataset_dicts[split_idx:]
        
        print(f"Split into {len(train_data)} training and {len(val_data)} validation images for {cell_type}")
        
        all_train_data.extend(train_data)
        all_val_data.extend(val_data)
    
    print(f"\nTotal dataset size:")
    print(f"Training: {len(all_train_data)} images")
    print(f"Validation: {len(all_val_data)} images")
    
    # Register the combined datasets
    DatasetCatalog.register("cell_train", lambda: all_train_data)
    MetadataCatalog.get("cell_train").set(thing_classes=CELL_TYPES)
    
    DatasetCatalog.register("cell_val", lambda: all_val_data)
    MetadataCatalog.get("cell_val").set(thing_classes=CELL_TYPES)
    
    print("Datasets registered: 'cell_train' and 'cell_val'")

def setup_predictor(cfg_path, model_weights):
    """Setup Detectron2 predictor"""
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set confidence threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    return predictor

def compute_per_class_accuracy(predictor, dataset_dicts, class_names):
    """Compute per-class accuracy"""
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for record in tqdm(dataset_dicts, desc="Evaluating"):
        image_path = record["file_name"]
        ground_truth = [ann["category_id"] for ann in record["annotations"]]
        
        # Run inference
        img = cv2.imread(image_path)
        if img is None:
            continue  # Skip if image is not readable
        outputs = predictor(img)
        pred_classes = outputs["instances"].pred_classes.tolist()
        
        # Handle cases where number of predictions and ground truths differ
        min_len = min(len(ground_truth), len(pred_classes))
        for i in range(min_len):
            gt = ground_truth[i]
            pred = pred_classes[i]
            class_total[gt] += 1
            if gt == pred:
                class_correct[gt] += 1
                
        # If there are more ground truths than predictions
        for gt in ground_truth[min_len:]:
            class_total[gt] += 1  # These are misses
        
        # If there are more predictions than ground truths, you might want to handle false positives
        # For simplicity, we are not counting false positives in accuracy
        
    # Calculate accuracy per class
    per_class_accuracy = {}
    for cls_idx, cls_name in enumerate(class_names):
        if class_total[cls_idx] > 0:
            accuracy = class_correct[cls_idx] / class_total[cls_idx]
            per_class_accuracy[cls_name] = accuracy
        else:
            per_class_accuracy[cls_name] = None  # No instances for this class
                
    return per_class_accuracy

def load_datasets(dataset_name):
    """Load dataset from DatasetCatalog"""
    dataset_dicts = DatasetCatalog.get(dataset_name)
    return dataset_dicts

def main():
    # Step 1: Register Datasets
    print("Registering datasets...")
    register_datasets()
    
    # Step 2: Load Validation Dataset
    dataset_name = "cell_val"
    try:
        dataset_dicts = load_datasets(dataset_name)
    except KeyError as e:
        print(f"Error: {e}")
        print("Available datasets:", DatasetCatalog.list())
        return
    
    # Step 3: Setup Predictor
    cfg_path = os.path.join(BASE_DIR, "configs", "detectron_configs", "mixed_detectron_config.yaml")
    model_weights = os.path.join(BASE_DIR, "src", "segmentation", "models", "mixed_detectron", "model_final.pth")
    
    if not os.path.exists(cfg_path):
        print(f"Configuration file not found at {cfg_path}")
        return
    
    if not os.path.exists(model_weights):
        print(f"Model weights not found at {model_weights}")
        return
    
    print("Setting up predictor...")
    predictor = setup_predictor(cfg_path, model_weights)
    
    # Step 4: Compute Per-Class Accuracy
    print("Computing per-class accuracy...")
    per_class_acc = compute_per_class_accuracy(predictor, dataset_dicts, CELL_TYPES)
    
    # Step 5: Display Results
    print("\nPer-Class Accuracy:")
    for cls, acc in per_class_acc.items():
        if acc is not None:
            print(f"{cls}: {acc * 100:.2f}%")
        else:
            print(f"{cls}: No instances in validation set.")

if __name__ == "__main__":
    main()
