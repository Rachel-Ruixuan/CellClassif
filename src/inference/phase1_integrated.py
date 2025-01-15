# phase1_integrated.py

import torch
import json
import os
import cv2
import gc
from typing import Dict, List
from tqdm import tqdm
import numpy as np
import math
import random
from datetime import datetime
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging

# Configure logging for the script
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler("phase1_integrated.log")  # Logs to a file
    ]
)

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
    - Visualizes tracking results by overlaying bounding boxes and track IDs.
    - Compiles annotated frames into a video.
    - Evaluates detection scores and saves metrics.
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

        # Initialize detection metrics storage
        self.detections = []  # List to store detection scores and class IDs

        # Define class mapping (adjust according to your model's classes)
        self.class_mapping = {
            0: "MDA",
            1: "FB",
            2: "M1",
            3: "M2"
        }

        # Initialize Detectron2
        logging.info("=== Phase 1: Initializing Detectron2 ===")
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

        logging.info("Building Detectron2 model...")
        try:
            # Build model explicitly
            model = build_model(self.detectron_cfg)
            model.eval()  # Set to evaluation mode

            logging.info("Loading Detectron2 model weights...")
            checkpointer = DetectionCheckpointer(model)
            checkpointer.load(mixed_detectron_path)  # Load weights only

            # Move model to device
            model.to(self.device)

            # Create predictor
            self.detectron_predictor = CustomPredictor(self.detectron_cfg, model)
            logging.info("Detectron2 model loaded successfully.")
        except Exception as e:
            logging.error(f"Error during Detectron2 initialization: {e}")
            raise

        # Initialize DeepSort tracker
        logging.info("Initializing DeepSort tracker...")
        self.tracker = DeepSort(
            max_age=50,
            n_init=2,
            nms_max_overlap=0.8,
            max_iou_distance=0.2,
            max_cosine_distance=0.4,
            # embedder='clip_RN50',    # More robust embedder
            embedder = 'mobilenet',
            half=True
        )
        logging.info("DeepSort tracker initialized.")

        # Initialize color mapping for track_ids
        self.track_colors = {}  # Dict to store unique colors for each track_id

    def process_video_frames(self, image_dir: str, inference_folder: str, visualize: bool = True):
        """
        Processes all frames in the specified directory.
        Saves segmentation results and tracking data as JSON files.
        Optionally visualizes tracking results by saving annotated frames.
        Compiles annotated frames into a video.
        Evaluates detection scores and saves metrics.
        """
        frame_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        logging.info(f"Found {len(frame_files)} frames in {image_dir}")

        # Initialize JSON files
        segmentation_output_file = os.path.join(inference_folder, "segmentation_results.json")
        tracking_output_file = os.path.join(inference_folder, "tracking_data.json")

        # Create or clear JSON files
        with open(segmentation_output_file, 'w') as f:
            pass  # Empty the file
        with open(tracking_output_file, 'w') as f:
            pass  # Empty the file

        # Create a directory for visualizations
        visualization_dir = os.path.join(inference_folder, "visualizations")
        if visualize:
            os.makedirs(visualization_dir, exist_ok=True)
            logging.info(f"Visualization frames will be saved to {visualization_dir}")

        # Process each frame
        for frame_idx, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
            frame_path = os.path.join(image_dir, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                logging.warning(f"Could not read frame {frame_file}. Skipping.")
                continue

            max_size = 1333
            height, width = frame.shape[:2]
            scale = min(max_size / max(height, width), 1.0)
            if scale < 1.0:
                new_height = int(height * scale)
                new_width = int(width * scale)
                frame = cv2.resize(frame, (new_width, new_height))
                logging.debug(f"Resized frame from {height}x{width} to {new_height}x{new_width}")
            else:
                new_height, new_width = height, width

            frame_results = self.process_single_frame(frame, frame_idx, inference_folder)

            # Visualize if enabled
            if visualize:
                annotated_frame = self.visualize_tracks(frame.copy(), frame_results, frame_idx)
                output_path = os.path.join(visualization_dir, f"visualization_{frame_idx:05d}.png")
                cv2.imwrite(output_path, annotated_frame)

            # Periodic memory management
            if frame_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            # Debug memory usage
            if self.device == 'cuda' and frame_idx % 5 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                logging.info(f"Frame {frame_idx} GPU memory: {allocated:.2f}GB")

        # After processing all frames, save tracking data
        with open(tracking_output_file, 'w') as f:
            json.dump(self.tracking_data, f, indent=4)
        logging.info(f"Tracking data saved to {tracking_output_file}")

        # Compile video if visualization is enabled
        if visualize:
            self.compile_video(visualization_dir, inference_folder)

        # Evaluate and save metrics
        self.evaluate_metrics(inference_folder)

    def process_single_frame(self, frame: np.ndarray, frame_idx: int, inference_folder: str) -> List[Dict]:
        """
        Processes a single frame: detection and tracking.
        Saves detection results to segmentation_output_file.
        Updates tracking_data.
        Accumulates detection scores and class IDs for evaluation.
        """
        logging.info(f"\nProcessing frame {frame_idx}")
        if self.device == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e9
            logging.debug(f"Initial GPU memory: {allocated:.2f}GB")

        # Resize frame if too large
        max_size = 1333
        height, width = frame.shape[:2]
        scale = min(max_size / max(height, width), 1.0)
        if scale < 1.0:
            new_height = int(height * scale)
            new_width = int(width * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            logging.debug(f"Resized frame from {height}x{width} to {new_height}x{new_width}")
        else:
            new_height, new_width = height, width

        with torch.no_grad():
            # Perform detection
            detectron_outputs = self.detectron_predictor(frame)

            if self.device == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1e9
                logging.debug(f"After detection GPU memory: {allocated:.2f}GB")

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

            # Debugging: Log number of detections
            logging.info(f"Number of detections in frame {frame_idx}: {len(detections)}")

            if len(detections) == 0:
                logging.info(f"No detections found in frame {frame_idx}. Skipping tracking.")
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
                        logging.warning(f"Invalid bbox format for detection: {bbox}")

                # Debugging: Log the type and structure of transformed_detections
                logging.debug(f"Type of transformed_detections: {type(transformed_detections)}")
                if len(transformed_detections) > 0:
                    logging.debug(f"Type of first detection: {type(transformed_detections[0])}")
                    logging.debug(f"Length of first detection: {len(transformed_detections[0])}")
                    logging.debug(f"First detection contents: {transformed_detections[0]}")

                # Update tracker
                tracks = self.tracker.update_tracks(transformed_detections, frame=frame)

                # Assign track IDs and accumulate tracking data
                for track in tracks:
                    if track.is_confirmed():
                        track_id = str(int(track.track_id))  # Convert to string for JSON compatibility
                        track_bbox = track.to_tlbr()  # [x1, y1, x2, y2]

                        # Find the detection in frame_results closest to this track's bbox
                        min_distance = float('inf')
                        closest_det_idx = -1
                        for idx, det in enumerate(frame_results):
                            det_bbox = det['bbox']
                            distance = math.hypot(
                                ((track_bbox[0] + track_bbox[2]) / 2) - ((det_bbox[0] + det_bbox[2]) / 2),
                                ((track_bbox[1] + track_bbox[3]) / 2) - ((det_bbox[1] + det_bbox[3]) / 2)
                            )
                            if distance < min_distance:
                                min_distance = distance
                                closest_det_idx = idx

                        SOME_THRESHOLD = 60.0

                        # Assign track_id to the closest detection if within threshold
                        if closest_det_idx != -1 and min_distance < SOME_THRESHOLD:
                            frame_results[closest_det_idx]["track_id"] = track_id

                            if track_id not in self.tracking_data:
                                self.tracking_data[track_id] = []

                            # Extract center coordinates from bbox
                            bbox = frame_results[closest_det_idx]['bbox']
                            center_x = (bbox[0] + bbox[2]) / 2
                            center_y = (bbox[1] + bbox[3]) / 2

                            self.tracking_data[track_id].append({
                                'frame': frame_idx,
                                'x': center_x,
                                'y': center_y,
                                'bbox': bbox,
                                'segmentation': frame_results[closest_det_idx]['segmentation'],
                                'score': frame_results[closest_det_idx]['score'],
                                'class_id': frame_results[closest_det_idx]['class_id']
                            })
                        else:
                            # Unassigned detection
                            if closest_det_idx != -1:
                                frame_results[closest_det_idx]["track_id"] = -1

            # Accumulate detection scores and class IDs for evaluation
            for det in frame_results:
                self.detections.append({
                    'score': det['score'],
                    'class_id': det['class_id']
                })

            # Save segmentation results to JSON
            segmentation_output_file = os.path.join(inference_folder, "segmentation_results.json")
            with open(segmentation_output_file, 'a') as f:
                json.dump({
                    "frame_idx": frame_idx,
                    "detections": frame_results
                }, f)
                f.write('\n')

            return frame_results

    def visualize_tracks(self, frame: np.ndarray, detections: List[Dict], frame_idx: int) -> np.ndarray:
        """
        Overlays bounding boxes, track IDs, and segmentation masks on the frame for visualization.
        - Unique color per track_id
        - Semi-transparent segmentation masks
        - Labels with type, score, and track_id (if tracked)
        - Untracked detections have semi-transparent white masks and appropriate labels
        """
        overlay = frame.copy()

        for det in detections:
            bbox = det['bbox']
            track_id = det.get('track_id', -1)
            class_id = det['class_id']
            class_label = self.class_mapping.get(class_id, "unknown")
            score = det['score']
            segmentation = det['segmentation']

            # Assign a unique color for each track_id
            if track_id != -1:
                if track_id not in self.track_colors:
                    # Generate a random color and store it
                    random.seed(int(track_id))
                    self.track_colors[track_id] = [random.randint(0, 255) for _ in range(3)]
                color = self.track_colors[track_id]
                mask_color = color
                alpha = 0.4  # Semi-transparent
            else:
                color = [255, 255, 255]  # White for untracked
                mask_color = [255, 255, 255]  # White
                alpha = 0.4  # Semi-transparent

            # Draw segmentation mask as a semi-transparent overlay
            if segmentation:
                try:
                    points = np.array(segmentation).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(overlay, [points], mask_color)
                except Exception as e:
                    logging.warning(f"Error drawing segmentation for frame {frame_idx}: {e}")

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Prepare label
            if track_id != -1:
                label = f"ID: {track_id} | {class_label} | Score: {score:.2f}"
            else:
                label = f"{class_label} | Score: {score:.2f}"

            # Determine label position and size
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(y1, label_size[1] + 10)

            # Put label text without background
            if track_id != -1:
                text_color = (color[0], color[1], color[2])  # Same as mask color
            else:
                text_color = (255, 255, 255)  # White text for untracked

            cv2.putText(frame, label, (x1 + 5, label_ymin - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Blend the overlay with the original frame to create semi-transparent masks
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def compile_video(self, visualization_dir: str, inference_folder: str, video_name: str = None, fps: int = 10):
        """
        Compiles annotated frames into a video file.
        Args:
            visualization_dir: Directory containing annotated frames.
            inference_folder: Directory to save the compiled video.
            video_name: Optional. Name of the output video file. If None, generates based on timestamp.
            fps: Frames per second for the output video.
        """
        if video_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = f"{self.video_id}_annotated_{timestamp}.mp4"

        video_path = os.path.join(inference_folder, video_name)

        # Get list of frame files sorted in order
        frame_files = sorted([f for f in os.listdir(visualization_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        if not frame_files:
            logging.error(f"No visualization frames found in {visualization_dir}. Cannot compile video.")
            return

        # Read the first frame to get frame dimensions
        first_frame_path = os.path.join(visualization_dir, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            logging.error(f"Cannot read the first frame {first_frame_path}. Cannot compile video.")
            return
        height, width, layers = first_frame.shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs if needed
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        logging.info(f"Compiling video at {video_path} with FPS={fps}")

        for frame_file in tqdm(frame_files, desc="Compiling video"):
            frame_path = os.path.join(visualization_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                logging.warning(f"Could not read frame {frame_file}. Skipping.")
                continue
            video_writer.write(frame)

        video_writer.release()
        logging.info(f"Video compilation completed successfully. Video saved at {video_path}")

    def evaluate_metrics(self, inference_folder: str):
        """
        Evaluates detection metrics such as average score, min/max scores, standard deviation,
        total number of detections, and detections per class.
        Saves the metrics as a JSON file in the inference_folder.
        """
        if not self.detections:
            avg_score = 0.0
            min_score = 0.0
            max_score = 0.0
            std_score = 0.0
            total_detections = 0
            class_counts = {}
            logging.info("No detections to evaluate.")
        else:
            scores = [d['score'] for d in self.detections]
            avg_score = float(np.mean(scores))
            min_score = float(np.min(scores))
            max_score = float(np.max(scores))
            std_score = float(np.std(scores))
            total_detections = len(scores)

            # Compute detections per class
            class_counts = {}
            for d in self.detections:
                class_label = self.class_mapping.get(d['class_id'], "unknown")
                class_counts[class_label] = class_counts.get(class_label, 0) + 1

            logging.info("Evaluation Metrics:")
            logging.info(f"Average Score: {avg_score:.4f}")
            logging.info(f"Minimum Score: {min_score:.4f}")
            logging.info(f"Maximum Score: {max_score:.4f}")
            logging.info(f"Standard Deviation: {std_score:.4f}")
            logging.info(f"Total Detections: {total_detections}")
            logging.info(f"Detections per Class: {class_counts}")

        metrics = {
            "average_score": avg_score,
            "min_score": min_score,
            "max_score": max_score,
            "std_score": std_score,
            "total_detections": total_detections,
            "detections_per_class": class_counts
        }

        metrics_file = os.path.join(inference_folder, "evaluation_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Evaluation metrics saved to {metrics_file}")

def main():
    BASE_DIR = "/scratch/rl4789/CellClassif"
    mixed_detectron_path = "/scratch/rl4789/CellClassif/src/segmentation/models/mixed_detectron/model_final.pth" 
    output_dir = "/scratch/rl4789/CellClassif/inference_results"
    cell_type = "M2"  # Assuming single category; adjust if multiple categories per video
    video_id = "m2-013"
    image_dir = os.path.join(BASE_DIR, "data", "raw", cell_type, video_id)
    inference_folder = os.path.join(output_dir, f"{video_id}_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(inference_folder, exist_ok=True)

    # Initialize Phase1_FrameProcessor
    phase1 = Phase1_FrameProcessor(
        mixed_detectron_path=mixed_detectron_path,
        output_dir=output_dir,
        cell_type=cell_type,
        video_id=video_id
    )

    # Process video frames with visualization
    phase1.process_video_frames(
        image_dir=image_dir,
        inference_folder=inference_folder,
        visualize=True
    )

    logging.info("Phase 1 processing completed successfully.")

if __name__ == "__main__":
    main()
