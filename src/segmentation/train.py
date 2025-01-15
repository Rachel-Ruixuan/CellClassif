# train.py

import torch
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os
import json
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, DatasetEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Configuration
BASE_DIR = "/scratch/rl4789/CellClassif"
CELL_TYPES = ["MDA", "FB", "M1", "M2"]  # All cell types for mixed training

class PerClassAccuracyEvaluator(DatasetEvaluator):
    """
    Custom evaluator to compute per-class classification accuracy alongside COCO metrics.
    """
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.output_dir = output_dir
        self.metadata = MetadataCatalog.get(dataset_name)
        
        # Initialize COCO ground truth
        self.coco_gt = COCO()
        self.coco_gt.dataset = self._load_ground_truth()
        self.coco_gt.createIndex()
        
        # Initialize COCO detections
        self.coco_dt = COCO()
        self.coco_dt.dataset = {"images": [], "annotations": [], "categories": self.coco_gt.dataset["categories"]}
        self.coco_dt.createIndex()
        
        # Initialize counters for per-class accuracy
        self.per_class_correct = defaultdict(int)
        self.per_class_total = defaultdict(int)
        self.iou_threshold = 0.5  # You can adjust this threshold as needed

    def _load_ground_truth(self):
        """
        Load ground truth annotations from the dataset.
        Assumes the dataset is registered in COCO format.
        """
        dataset_dicts = DatasetCatalog.get(self.dataset_name)
        coco_gt = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": cls} for i, cls in enumerate(self.metadata.thing_classes)]
        }
        annotation_id = 1
        for idx, record in enumerate(dataset_dicts):
            image_id = idx + 1
            coco_gt["images"].append({
                "id": image_id,
                "file_name": os.path.basename(record["file_name"]),
                "height": record["height"],
                "width": record["width"]
            })
            for ann in record["annotations"]:
                coco_gt["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": ann["iscrowd"],
                    "segmentation": ann["segmentation"]
                })
                annotation_id += 1
        return coco_gt

    def reset(self):
        self.coco_dt.dataset["images"] = []
        self.coco_dt.dataset["annotations"] = []

    def process(self, inputs, outputs):
        """
        Processes a single batch of inputs and outputs.
        Collects predictions to compute per-class accuracy.
        """
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            img_annotations = output["instances"].to("cpu")
            boxes = img_annotations.pred_boxes.tensor.numpy()
            classes = img_annotations.pred_classes.numpy()
            scores = img_annotations.scores.numpy()

            # Assign a unique ID for each prediction
            pred_id_start = len(self.coco_dt.dataset["annotations"]) + 1
            for i in range(len(boxes)):
                bbox = boxes[i].tolist()
                x1, y1, x2, y2 = bbox
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                area = bbox_width * bbox_height  # Compute area

                pred_annotation = {
                    "id": pred_id_start + i,
                    "image_id": image_id,
                    "category_id": int(classes[i]),
                    "bbox": [x1, y1, bbox_width, bbox_height],
                    "score": float(scores[i]),
                    "area": float(area),  # **Include 'area' key**
                    # "segmentation": [],  # Optional: Add segmentation if needed
                }
                self.coco_dt.dataset["annotations"].append(pred_annotation)

    def evaluate(self):
        """
        Computes per-class accuracy and returns evaluation results.
        """
        # Load ground truth and detections into COCO format
        self.coco_gt.createIndex()
        self.coco_dt.createIndex()

        # Initialize COCOeval
        coco_eval = COCOeval(self.coco_gt, self.coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Compute per-class accuracy
        # Initialize a mapping from image_id to annotations
        gt_ann_map = defaultdict(list)
        for ann in self.coco_gt.dataset["annotations"]:
            gt_ann_map[ann["image_id"]].append(ann)

        dt_ann_map = defaultdict(list)
        for ann in self.coco_dt.dataset["annotations"]:
            dt_ann_map[ann["image_id"]].append(ann)

        for image_id in self.coco_gt.getImgIds():
            gt_anns = gt_ann_map[image_id]
            dt_anns = dt_ann_map[image_id]

            gt_boxes = [ann["bbox"] for ann in gt_anns]
            gt_classes = [ann["category_id"] for ann in gt_anns]
            gt_matched = [False] * len(gt_anns)

            dt_boxes = [ann["bbox"] for ann in dt_anns]
            dt_classes = [ann["category_id"] for ann in dt_anns]

            # Convert to [x1, y1, x2, y2]
            gt_boxes_xyxy = [[x, y, x + w, y + h] for (x, y, w, h) in gt_boxes]
            dt_boxes_xyxy = [[x, y, x + w, y + h] for (x, y, w, h) in dt_boxes]

            # Compute IoU matrix
            ious = compute_iou_matrix(dt_boxes_xyxy, gt_boxes_xyxy)

            for dt_idx, dt_class in enumerate(dt_classes):
                if len(gt_boxes_xyxy) == 0:
                    continue  # No ground truths to match
                best_gt_idx = np.argmax(ious[dt_idx])
                best_iou = ious[dt_idx][best_gt_idx]

                if best_iou >= self.iou_threshold and not gt_matched[best_gt_idx]:
                    gt_matched[best_gt_idx] = True
                    gt_class = gt_classes[best_gt_idx]
                    self.per_class_total[gt_class] += 1
                    if dt_class == gt_class:
                        self.per_class_correct[gt_class] += 1
                else:
                    # False Positive: prediction did not match any ground truth
                    pass  # Ignored in accuracy calculation

            # Handle unmatched ground truths (missed detections)
            for gt_idx, matched in enumerate(gt_matched):
                if not matched:
                    gt_class = gt_classes[gt_idx]
                    self.per_class_total[gt_class] += 1  # Missed instance

        # Calculate per-class accuracy
        per_class_accuracy = {}
        for cat in self.coco_gt.dataset["categories"]:
            cat_id = cat["id"]
            cat_name = cat["name"]
            correct = self.per_class_correct.get(cat_id, 0)
            total = self.per_class_total.get(cat_id, 0)
            if total > 0:
                accuracy = (correct / total) * 100
                per_class_accuracy[cat_name] = accuracy
            else:
                per_class_accuracy[cat_name] = None  # No instances for this class

        # Print per-class accuracy
        print("\nPer-Class Classification Accuracy:")
        for cls, acc in per_class_accuracy.items():
            if acc is not None:
                print(f"{cls}: {acc:.2f}%")
            else:
                print(f"{cls}: No instances in validation set.")

        return {
            "bbox": coco_eval.stats[:6],
            "per_class_accuracy": per_class_accuracy
        }

def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Each box is [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea

    if unionArea == 0:
        return 0.0
    else:
        return interArea / unionArea

def compute_iou_matrix(boxes1, boxes2):
    """
    Compute IoU matrix between two lists of boxes.
    boxes1: List of [x1, y1, x2, y2]
    boxes2: List of [x1, y1, x2, y2]
    Returns:
        iou: numpy array of shape (len(boxes1), len(boxes2))
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])

    inter_width = np.maximum(0, inter_x2 - inter_x1)
    inter_height = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union_area = area1[:, None] + area2 - inter_area
    iou = inter_area / union_area

    return iou

class MixedTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return PerClassAccuracyEvaluator(dataset_name, cfg, False, output_folder)

def load_pseudo_ground_truth(cell_type):
    """Load predictions from specialized Detectron models as pseudo ground truth."""
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
                "image_id": len(dataset_dicts) + 1,  # Ensure unique image IDs
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
                if not pred.get("segmentation") or not isinstance(pred["segmentation"][0], list):
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
                    "area": bbox_width * bbox_height  # Compute area
                }
                valid_annotations.append(annotation)
            
            if valid_annotations:  # Only add images with valid predictions
                record["annotations"] = valid_annotations
                dataset_dicts.append(record)
    
    print(f"Loaded {len(dataset_dicts)} valid images for {cell_type}")
    return dataset_dicts

def register_datasets():
    """Register datasets from pseudo ground truth."""
    # **Do NOT attempt to remove existing datasets.**
    # If you need to overwrite, ensure unique dataset names or run in a fresh environment.
    
    all_train_data = []
    all_val_data = []
    
    for cell_type in CELL_TYPES:
        print(f"\nProcessing {cell_type} dataset...")
        dataset_dicts = load_pseudo_ground_truth(cell_type)
        
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
    
    return "cell_train", "cell_val"

def setup_cfg(train_dataset, val_dataset):
    """Set up the configuration for training."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (val_dataset,)
    
    # Model configuration
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )  # Initialize from COCO pre-trained weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CELL_TYPES)  # 4 classes
    
    # Adjusted training parameters
    cfg.SOLVER.IMS_PER_BATCH = 4  # Batch size per image
    cfg.SOLVER.BASE_LR = 0.0005   # Learning rate
    cfg.SOLVER.MAX_ITER = 30000   # Total training iterations
    cfg.SOLVER.STEPS = (20000, 25000)  # Learning rate decay steps
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Save checkpoint every 1000 iterations
    
    # Adjusted model parameters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Number of regions per image
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7    # Confidence threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3      # Non-Maximum Suppression threshold
    
    # Augmentation configuration
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Output directory
    cfg.OUTPUT_DIR = os.path.join(BASE_DIR, "src", "segmentation", "models", "mixed_detectron")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Other configurations
    cfg.DATALOADER.NUM_WORKERS = 4  # Number of data loading workers
    
    return cfg

def visualize_samples(dataset_name, num_samples=5):
    """Visualize sample annotations to verify correctness."""
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    for d in dataset_dicts[:num_samples]:
        img = cv2.imread(d["file_name"])
        if img is None:
            continue
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
        vis = visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(12, 12))
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.axis('off')
        plt.show()

def main():
    # Initialize logger
    setup_logger()
    
    print("Registering datasets from pseudo ground truth...")
    train_dataset, val_dataset = register_datasets()
    
    # Optional: Visualize some training samples to verify annotations
    print("\nVisualizing sample training annotations...")
    visualize_samples(train_dataset, num_samples=3)
    
    print("\nConfiguring model...")
    cfg = setup_cfg(train_dataset, val_dataset)
    
    print("\nStarting training...")
    trainer = MixedTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    print("\nEvaluating model...")
    # Build test loader
    val_loader = build_detection_test_loader(cfg, val_dataset)
    # Create evaluator
    evaluator = PerClassAccuracyEvaluator(val_dataset, cfg, False, output_dir=cfg.OUTPUT_DIR)
    # Run inference and evaluation
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    # Print detailed evaluation results
    print("\nDetailed Evaluation Results:")
    print("\nBounding Box Detection Metrics:")
    if 'bbox' in results:
        bbox_metrics = results['bbox']
        metrics_names = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large']
        for name, value in zip(metrics_names, bbox_metrics):
            print(f"{name}: {value:.3f}")
    
    print("\nPer-Class Classification Accuracy:")
    if 'per_class_accuracy' in results:
        per_class_acc = results['per_class_accuracy']
        for cls, acc in per_class_acc.items():
            if acc is not None:
                print(f"{cls}: {acc:.2f}%")
            else:
                print(f"{cls}: No instances in validation set.")

if __name__ == "__main__":
    main()
