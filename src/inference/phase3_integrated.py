# phase3_integrated.py

import json
import os
import logging
import cv2
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler("phase3_integrated.log")  # Logs to a file
    ]
)

class Phase3_EnsemblerAndVisualizer:
    """
    Phase 3: Ensembling, Visualization, and Evaluation
    - Combines detectron's detection scores with motion model's class probabilities.
    - Assigns final class labels based on the highest ensemble score.
    - Outputs evaluation metrics: total_detections and detections_per_class.
    - Annotates frames with final predictions.
    - Creates a final annotated video.
    """

    def __init__(
        self, 
        class_mapping: dict,
        output_dir: str,
        video_id: str,
        inference_folder: str,
        image_dir: str
    ):
        self.class_mapping = class_mapping  # e.g., {0: "MDA", 1: "FB", 2: "M1", 3: "M2"}
        self.output_dir = output_dir
        self.video_id = video_id
        self.inference_folder = inference_folder
        self.image_dir = image_dir
        self.annotated_frames_dir = os.path.join(self.inference_folder, "annotated_frames")
        os.makedirs(self.annotated_frames_dir, exist_ok=True)

        # Preload sorted frame filenames for mapping
        self.frame_files_sorted = self.get_sorted_frame_files()
        if not self.frame_files_sorted:
            logging.error(f"No frame files found in {self.image_dir}. Exiting.")
            exit(1)
        else:
            logging.info(f"Found {len(self.frame_files_sorted)} frame files in {self.image_dir}.")

    def get_sorted_frame_files(self) -> list:
        """
        Retrieves and sorts all frame files in the image directory.
        Sorting is based on frame numbering inferred from filenames.
        Adjust the sorting mechanism if your filenames have a different pattern.
        """
        frame_files = [f for f in os.listdir(self.image_dir) 
                      if f.endswith('.jpg') or f.endswith('.png')]
        if not frame_files:
            return []

        # Sort the frame files in ascending order
        # Assuming filenames like 'm2-013_t03_ch00.jpg', sorting lexically should work
        frame_files_sorted = sorted(frame_files)
        return frame_files_sorted

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
                if line.strip():  # Ensure line is not empty
                    data = json.loads(line)
                    segmentation_results.append(data)
        logging.info(f"Loaded segmentation results from {segmentation_output_file}")
        return segmentation_results

    def perform_ensembling(self, segmentation_results: list, motion_features: dict) -> dict:
        """
        Combines detectron's detection scores with motion model's class probabilities.
        Assigns the final class based on the highest ensemble score.
        Saves final predictions and detection summary.
        """
        final_predictions = {}
        detections_per_class = {
            "M1": 0,
            "M2": 0,
            "MDA": 0,
            "FB": 0
        }
        total_detections = 0

        # Define weights (adjust as needed)
        WEIGHT_DETECTRON = 0.6
        WEIGHT_MOTION = 0.4

        logging.info("Starting ensembling process...")

        for detection in tqdm(segmentation_results, desc="Ensembling detections"):
            frame_idx = detection.get("frame_idx")
            detections = detection.get("detections", [])
            for det in detections:
                track_id = det.get("track_id", -1)
                if track_id == -1:
                    continue  # Skip unconfirmed tracks

                # Retrieve detectron's detection score
                detectron_score = det.get("score", 0.0)

                # Retrieve detectron's class_id
                detectron_class_id = det.get("class_id", -1)
                if detectron_class_id not in self.class_mapping:
                    logging.warning(f"Invalid class_id {detectron_class_id} for track ID {track_id}. Skipping.")
                    continue
                detectron_class_label = self.class_mapping[detectron_class_id]

                # Create detectron_probs as one-hot encoding with detection score
                detectron_probs = [0.0] * len(self.class_mapping)
                detectron_probs[detectron_class_id] = detectron_score

                # Retrieve motion model's class probabilities
                motion = motion_features.get(str(track_id), {})
                motion_probs = motion.get("probabilities", [])
                if len(motion_probs) != len(self.class_mapping):
                    logging.warning(f"Track ID {track_id} has invalid motion probabilities length ({len(motion_probs)}). Expected {len(self.class_mapping)}. Skipping.")
                    continue

                # Ensembling: Weighted sum of detectron and motion probabilities
                ensemble_scores = {}
                for idx, class_label in self.class_mapping.items():
                    detectron_prob = detectron_probs[idx] if idx < len(detectron_probs) else 0.0
                    motion_prob = motion_probs[idx] if idx < len(motion_probs) else 0.0
                    ensemble_score = WEIGHT_DETECTRON * detectron_prob + WEIGHT_MOTION * motion_prob
                    ensemble_scores[class_label] = ensemble_score

                # Assign the class with the highest ensemble score
                final_class = max(ensemble_scores, key=ensemble_scores.get)
                final_score = ensemble_scores[final_class]

                # Update final predictions
                final_predictions[track_id] = {
                    "frame_idx": frame_idx,
                    "bbox": det.get("bbox", []),
                    "score": final_score,
                    "class_id": self.get_class_id(final_class),
                    "class_label": final_class,
                    "segmentation": det.get("segmentation", []),
                    "ensemble_scores": ensemble_scores
                }

                # Update summary
                if final_class in detections_per_class:
                    detections_per_class[final_class] += 1
                else:
                    detections_per_class[final_class] = 1
                total_detections += 1

        # Save final predictions
        final_predictions_file = os.path.join(self.inference_folder, "final_predictions.json")
        try:
            with open(final_predictions_file, 'w') as f:
                json.dump(final_predictions, f, indent=4)
            logging.info(f"Final predictions saved to {final_predictions_file}")
        except Exception as e:
            logging.error(f"Error saving final predictions: {e}")
            raise

        # Save detection summary
        detection_summary = {
            "total_detections": total_detections,
            "detections_per_class": detections_per_class
        }
        detection_summary_file = os.path.join(self.inference_folder, "detection_summary.json")
        try:
            with open(detection_summary_file, 'w') as f:
                json.dump(detection_summary, f, indent=4)
            logging.info(f"Detection summary saved to {detection_summary_file}")
        except Exception as e:
            logging.error(f"Error saving detection summary JSON: {e}")
            raise

        logging.info("Ensembling process completed.")
        return final_predictions

    def get_class_id(self, class_label: str) -> int:
        """
        Maps class_label to class_id using predefined mapping.
        """
        label_to_id = {label: idx for idx, label in self.class_mapping.items()}
        return label_to_id.get(class_label, -1)

    def annotate_frames(self, final_predictions: dict):
        """
        Annotates frames with final predictions.
        Saves annotated frames to 'annotated_frames' directory.
        """
        logging.info("Starting frame annotation...")

        for track_id, pred in tqdm(final_predictions.items(), desc="Annotating tracks"):
            frame_idx = pred.get("frame_idx")
            bbox = pred.get("bbox", [])
            class_label = pred.get("class_label", "unknown")
            score = pred.get("score", 0.0)

            if len(bbox) != 4:
                logging.warning(f"Invalid bbox for track ID {track_id}. Skipping annotation.")
                continue

            # Map frame_idx to frame_file using sorted frame list
            if frame_idx < 0 or frame_idx >= len(self.frame_files_sorted):
                logging.warning(f"Frame index {frame_idx} for track ID {track_id} is out of bounds. Skipping annotation.")
                continue

            frame_file = self.frame_files_sorted[frame_idx]
            frame_path = os.path.join(self.image_dir, frame_file)

            # Read the frame image
            frame = cv2.imread(frame_path)
            if frame is None:
                logging.warning(f"Could not read frame {frame_file}. Skipping annotation.")
                continue

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label text
            label = f"{class_label}: {score:.2f}"

            # Calculate label size
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(y1, label_size[1] + 10)

            # Draw label background
            cv2.rectangle(frame, (x1, y_label - label_size[1] - 10), 
                                 (x1 + label_size[0] + 10, y_label + base_line - 10), 
                                 (0, 255, 0), cv2.FILLED)
            # Put label text
            cv2.putText(frame, label, (x1 + 5, y_label - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Save the annotated frame
            annotated_frame_path = os.path.join(self.annotated_frames_dir, frame_file)
            try:
                cv2.imwrite(annotated_frame_path, frame)
            except Exception as e:
                logging.error(f"Error saving annotated frame {frame_file}: {e}")
                continue

        logging.info(f"Annotated frames saved to {self.annotated_frames_dir}")

    def create_final_video(self, output_video_path: str, fps: int = 30):
        """
        Creates a video from annotated frames.
        """
        logging.info("Starting video creation...")

        # Get list of annotated frame files
        frame_files = sorted([
            f for f in os.listdir(self.annotated_frames_dir) 
            if f.endswith('.jpg') or f.endswith('.png')
        ])

        if not frame_files:
            logging.error(f"No annotated frames found in {self.annotated_frames_dir}. Cannot create video.")
            return

        # Read the first frame to get frame size
        first_frame_path = os.path.join(self.annotated_frames_dir, frame_files[0])
        frame = cv2.imread(first_frame_path)
        if frame is None:
            logging.error(f"Could not read the first frame {frame_files[0]}. Cannot create video.")
            return

        height, width, layers = frame.shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for frame_file in tqdm(frame_files, desc="Creating video"):
            frame_path = os.path.join(self.annotated_frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                logging.warning(f"Could not read annotated frame {frame_file}. Skipping.")
                continue
            video.write(frame)

        video.release()
        logging.info(f"Final video saved to {output_video_path}")

def main():
    """
    Main function to execute Phase 3: Ensembling, Visualization, and Evaluation.
    """
    # Hardcoded paths and parameters
    BASE_DIR = "/scratch/rl4789/CellClassif"
    video_id = "m2-013"
    output_dir = "/scratch/rl4789/CellClassif/inference_results/m2-013_inference_20241230_053002"
    inference_folder = output_dir  # Assuming detection_summary and final_predictions are in output_dir
    image_dir = "/scratch/rl4789/CellClassif/data/raw/M2/m2-013"  # Adjust as per your directory structure

    # Define class mapping
    class_mapping = {
        0: "MDA",
        1: "FB",
        2: "M1",
        3: "M2"
    }

    # Initialize Phase3_EnsemblerAndVisualizer
    phase3 = Phase3_EnsemblerAndVisualizer(
        class_mapping=class_mapping,
        output_dir=output_dir,
        video_id=video_id,
        inference_folder=inference_folder,
        image_dir=image_dir
    )

    # Load motion features and segmentation results
    motion_features = phase3.load_motion_predictions()
    if not motion_features:
        logging.error("No motion features loaded. Exiting.")
        exit(1)

    segmentation_results = phase3.load_segmentation_results()
    if not segmentation_results:
        logging.error("No segmentation results loaded. Exiting.")
        exit(1)

    # Perform ensembling
    final_predictions = phase3.perform_ensembling(segmentation_results, motion_features)
    if not final_predictions:
        logging.warning("No final predictions generated.")
    else:
        logging.info(f"Generated {len(final_predictions)} final predictions.")

    # Annotate frames
    phase3.annotate_frames(final_predictions)

    # Create final video
    final_video_path = os.path.join(output_dir, f"{video_id}_prediction_final.mp4")
    phase3.create_final_video(final_video_path)

    logging.info("Phase 3 processing completed successfully.")

if __name__ == "__main__":
    main()
