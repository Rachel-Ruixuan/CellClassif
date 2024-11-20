# 1. Imports and Setup
import torch
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os
import json
import cv2
import random
import time
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import matplotlib.pyplot as plt
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
# Add these new imports
from detectron2.engine import HookBase
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetMapper
from detectron2.utils import comm

# CUDA Detection
print("\nChecking CUDA availability:")
if torch.cuda.is_available():
    print(f"CUDA is available:")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    # Set default device
    torch.cuda.set_device(0)
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
else:
    print("CUDA is not available. Training will proceed on CPU.")
    print("Warning: Training on CPU may be significantly slower.")


# 2. Path Configuration
BASE_DIR = "/scratch/rl4789/CellClassif"
# CELL_TYPES = ["MDA", "FB", "ADSC"]
CELL_TYPES = ["M2"]

for cell_type in CELL_TYPES:
    register_coco_instances(
        f"{cell_type}_train",
        {},
        os.path.join(BASE_DIR, "data/processed/detectron_format", cell_type, f"train/{cell_type}_coco.json"),
        os.path.join(BASE_DIR, "data/processed/detectron_format", cell_type, f"train/images")
    )
    register_coco_instances(
        f"{cell_type}_val",
        {},
        os.path.join(BASE_DIR, "data/processed/detectron_format", cell_type, f"val/{cell_type}_val_coco.json"),
        os.path.join(BASE_DIR, "data/processed/detectron_format", cell_type, "val/images")
    )

# 3. Custom Trainer with Validation Losss
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
        
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
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

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks

# 4. Training Configuration and Execution
def train_cell_type(cell_type):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (f"{cell_type}_train",)
    cfg.DATASETS.TEST = (f"{cell_type}_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 10000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class per cell type
    cfg.OUTPUT_DIR = os.path.join(BASE_DIR, "models", cell_type)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Set device based on CUDA availability
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    return cfg

# 6. Main Execution
def main():
    for cell_type in CELL_TYPES:
        print(f"Training {cell_type} model...")
        cfg = train_cell_type(cell_type)

if __name__ == "__main__":
    setup_logger()
    main()