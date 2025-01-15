import torch
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os
import json
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import HookBase
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetMapper
from detectron2.utils import comm
import shutil
from detectron2.structures import BoxMode
import glob
from tqdm import tqdm

# Configuration
BASE_DIR = "/scratch/rl4789/CellClassif"
CELL_TYPES = ["MDA", "FB", "M1", "M2"]  # All cell types for mixed training

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
        
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()
        return losses
            
    def _get_loss(self, data):
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        return sum(loss for loss in metrics_dict.values())
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter == self.trainer.max_iter or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()


class MixedTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = COCOEvaluator
        evaluator_list.append(evaluator_type(
            dataset_name, 
            output_dir=output_folder,
            tasks=("bbox", "segm"),
        ))
        return evaluator_list[0]

def evaluate_model(cfg, model):
    """
    Evaluate the model on the validation set
    """
    evaluator = MixedTrainer.build_evaluator(
        cfg, 
        cfg.DATASETS.TEST[0],
        os.path.join(cfg.OUTPUT_DIR, "inference")
    )
    
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    results = inference_on_dataset(model, val_loader, evaluator)
    verify_results(results)
    return results


def load_specialized_predictions(cell_type):
    """Load predictions from specialized Detectron models with improved filtering"""
    predictions_base = os.path.join(BASE_DIR, "data/interim/segmentation", cell_type, "predictions")
    raw_images_base = os.path.join(BASE_DIR, "data/raw", cell_type)
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
    # Clear existing registrations
    for d in ["cell_train", "cell_val"]:
        if d in DatasetCatalog:
            DatasetCatalog.remove(d)
    
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
        
        print(f"Split into {len(train_data)} training and {len(val_data)} validation images")
        
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
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CELL_TYPES)
    
    # Adjusted training parameters
    cfg.SOLVER.IMS_PER_BATCH = 4  # Reduced batch size
    cfg.SOLVER.BASE_LR = 0.0005   # Reduced learning rate
    cfg.SOLVER.MAX_ITER = 30000   # Increased iterations
    cfg.SOLVER.STEPS = (20000, 25000)  # Adjusted steps
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    
    # Adjusted model parameters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Reduced batch size per image
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7    # Increased confidence threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3      # Adjusted NMS threshold

    cfg.TEST.EVAL_PERIOD = 5000  # Evaluate every 5000 iterations
    
    # Augmentation configuration
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    cfg.OUTPUT_DIR = os.path.join(BASE_DIR, "src", "segmentation", "models", "mixed_detectron")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    return cfg

def main():
    setup_logger()
    
    print("Registering datasets from specialized predictions...")
    train_dataset, val_dataset = register_datasets()
    
    print("\nConfiguring model...")
    cfg = setup_cfg(train_dataset, val_dataset)
    
    print("\nStarting training...")
    trainer = MixedTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    print("\nEvaluating model...")
    results = evaluate_model(cfg, trainer.model)
    
    # Print detailed evaluation results
    print("\nDetailed Evaluation Results:")
    print("\nBounding Box Detection:")
    for k, v in results['bbox'].items():
        print(f"{k}: {v:.3f}")
    
    print("\nSegmentation:")
    for k, v in results['segm'].items():
        print(f"{k}: {v:.3f}")
    
    # Print per-class results
    print("\nPer-class AP50 (Bounding Box):")
    for cls_idx, cls_name in enumerate(CELL_TYPES):
        if f'AP50-{cls_name}' in results['bbox']:
            print(f"{cls_name}: {results['bbox'][f'AP50-{cls_name}']:.3f}")
    
    print("\nPer-class AP50 (Segmentation):")
    for cls_idx, cls_name in enumerate(CELL_TYPES):
        if f'AP50-{cls_name}' in results['segm']:
            print(f"{cls_name}: {results['segm'][f'AP50-{cls_name}']:.3f}")

if __name__ == "__main__":
    main()