# phase3.py

import torch
import json
import os
import cv2
from typing import Dict, List
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler("phase3.log")  # Logs to a file
    ]
)

class Phase3_EnsemblerAndVisualizer:
    """
    Phase 3: Ensembling & Visualization
    - Performs ensembling of detection and motion features.
    - Annotates frames with combined predictions.
    - Generates a final annotated video.
    """

    def __init__(
        self, 
        class_mapping: dict,
        output_dir: str,
        video_id: str,
        inference_folder: str
    ):
        self.class_mapping = class_mapping
        self.output_dir = output_dir
        self.video_id = video_id
        self.inference_folder = inference_folder
        os.makedirs(output_dir, exist_ok=True)

    def load_motion_predictions(self) -> dict:
        """
        Loads motion features from 'motion_features.json'.
        Returns a dictionary of motion features.
        """
        motion_output_file = os.path.join(self.inference_folder, "motion_features.json")
        if not os.path.exists(motion_output_file):
            logging.error(f"Motion features file {motion_output_file} does not exist.")
            return {}

        with open(motion_output_file, 'r') as f:
            motion_features = json.load(f)
        logging.info(f"Loaded motion features from {motion_output_file}")
        return motion_features

    def load_segmentation_results(self) -> list:
        """
        Loads segmentation results from 'segmentation_results.json'.
        Returns a list of detection results per frame.
        """
        segmentation_output_file = os.path.join(self.inference_folder, "segmentation_results.json")
        if not os.path.exists(segmentation_output_file):
            logging.error(f"Segmentation results file {segmentation_output_file} does not exist.")
            return []

        segmentation_results = []
        with open(segmentation_output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                segmentation_results.append(data)
        logging.info(f"Loaded segmentation results from {segmentation_output_file}")
        return segmentation_results

    def perform_ensembling(self, segmentation_results: list, motion_features: dict) -> dict:
        """
        Combines detection and motion features.
        Returns a dictionary of final predictions.
        """
        final_predictions = {}
        logging.info("=== Phase 3: Performing Ensembling ===")

        for detection in segmentation_results:
            frame_idx = detection["frame_idx"]
            for det in detection["detections"]:
                track_id = det.get("track_id", -1)
                if track_id == -1:
                    continue  # Skip unconfirmed tracks

                # Retrieve motion features
                motion = motion_features.get(track_id, {})
                average_speed = motion.get("average_speed", 0.0)
                average_acceleration = motion.get("average_acceleration", 0.0)
                predicted_class = motion.get("predicted_class", -1)

                # Combine detection score with motion features (example logic)
                # Adjust the ensembling strategy as per your requirements
                combined_score = det["score"] * (1 + average_speed + abs(average_acceleration))  # Simple example

                # Aggregate final predictions
                final_predictions[track_id] = {
                    "frame_idx": frame_idx,
                    "bbox": det["bbox"],
                    "score": combined_score,
                    "class_id": det["class_id"],
                    "class_label": self.class_mapping.get(det["class_id"], "unknown"),
                    "segmentation": det["segmentation"],
                    "average_speed": average_speed,
                    "average_acceleration": average_acceleration,
                    "predicted_class": predicted_class
                }

        # Save final predictions to JSON
        final_predictions_file = os.path.join(self.inference_folder, "final_predictions.json")
        try:
            with open(final_predictions_file, 'w') as f:
                json.dump(final_predictions, f, indent=4)
            logging.info(f"Final predictions saved to {final_predictions_file}")
        except Exception as e:
            logging.error(f"Error saving final predictions: {e}")
            raise

        return final_predictions

    def annotate_frames(self, final_predictions: dict, image_dir: str):
        """
        Annotates frames with final predictions.
        Saves annotated frames to 'annotated_frames' directory.
        """
        annotated_frames_dir = os.path.join(self.inference_folder, "annotated_frames")
        os.makedirs(annotated_frames_dir, exist_ok=True)

        logging.info("=== Phase 3: Annotating Frames ===")

        # Load segmentation results to get frame-wise detections
        segmentation_results = self.load_segmentation_results()

        for detection in tqdm(segmentation_results, desc="Annotating frames"):
            frame_idx = detection["frame_idx"]
            frame_file = f"frame_{frame_idx:04d}.jpg"  # Adjust naming convention as per your frames
            frame_path = os.path.join(image_dir, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                logging.warning(f"Could not read frame {frame_file}. Skipping annotation.")
                continue

            for det in detection["detections"]:
                track_id = det.get("track_id", -1)
                if track_id == -1:
                    continue  # Skip unconfirmed tracks

                final_pred = final_predictions.get(track_id, {})
                if not final_pred:
                    continue

                bbox = final_pred["bbox"]
                score = final_pred["score"]
                class_label = final_pred["class_label"]
                predicted_class = final_pred.get("predicted_class", -1)

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Example label: class label and predicted class
                label = f"{class_label}: {score:.2f}, Pred: {predicted_class}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save annotated frame
            annotated_frame_path = os.path.join(annotated_frames_dir, frame_file)
            try:
                cv2.imwrite(annotated_frame_path, frame)
            except Exception as e:
                logging.error(f"Error saving annotated frame {frame_file}: {e}")
                continue

        logging.info(f"Annotated frames saved to {annotated_frames_dir}")

    def create_final_video(self, annotated_frames_dir: str, output_video_path: str, fps: int = 30):
        """
        Creates a video from annotated frames.
        """
        logging.info("=== Phase 3: Creating Final Video ===")

        frame_files = sorted([f for f in os.listdir(annotated_frames_dir) if f.endswith('.jpg') or f.endswith('.png')])

        if not frame_files:
            logging.error(f"No annotated frames found in {annotated_frames_dir}. Cannot create video.")
            return

        # Read the first frame to get frame size
        first_frame_path = os.path.join(annotated_frames_dir, frame_files[0])
        frame = cv2.imread(first_frame_path)
        if frame is None:
            logging.error(f"Could not read the first frame {frame_files[0]}. Cannot create video.")
            return

        height, width, layers = frame.shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for frame_file in tqdm(frame_files, desc="Creating video"):
            frame_path = os.path.join(annotated_frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                logging.warning(f"Could not read annotated frame {frame_file}. Skipping.")
                continue
            video.write(frame)

        video.release()
        logging.info(f"Final video saved to {output_video_path}")
