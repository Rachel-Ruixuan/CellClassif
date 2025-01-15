# phase2_integrated.py

import torch
import json
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict
from tqdm import tqdm
from models import MotionFeatureNet  # Ensure this import is correct

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler("phase2_integrated.log")  # Logs to a file
    ]
)

class Phase2_MotionFeatureExtractor:
    """
    Phase 2: Motion Feature Extraction
    - Extracts motion-related features from tracking data.
    - Processes them through the MotionFeatureNet model.
    - Saves motion features as JSON and CSV files.
    - Outputs total detections and detections per class as JSON.
    """

    def __init__(
        self, 
        motion_model_config_path: str,
        output_dir: str,
        cell_type: str,
        video_id: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.motion_model_config_path = motion_model_config_path
        self.output_dir = output_dir
        self.cell_type = cell_type
        self.video_id = video_id
        os.makedirs(output_dir, exist_ok=True)

        # Initialize Motion Feature Model
        logging.info("Loading Motion Feature Model...")
        try:
            # Load config from JSON file
            config_path = motion_model_config_path
            with open(config_path, 'r') as f:
                config = json.load(f)  # Assuming the config is stored in JSON

            # Extract model parameters from config
            input_dim = config['model']['input_dim']
            num_classes = config['model']['num_classes']
            weights_path = config['model']['weights_path']
            hidden_dims = config['model']['hidden_dims']
            dropout_rate = config['model']['dropout_rate']

            # Initialize the model with correct input_dim and num_classes
            self.motion_model = MotionFeatureNet(input_dim, num_classes, config).to(self.device)

            # Load model weights and scaler
            checkpoint = torch.load(weights_path, map_location=self.device)

            # Determine if checkpoint contains 'model_state_dict' and 'scaler'
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint  # Assume the checkpoint is the state_dict itself

            self.motion_model.load_state_dict(state_dict)
            self.motion_model.eval()  # Set to evaluation mode
            logging.info("Motion Feature Model loaded successfully.")

            # Load scaler
            if 'scaler' in checkpoint:
                self.scaler = checkpoint['scaler']
                logging.info("Scaler loaded successfully.")
            else:
                raise KeyError("Scaler not found in the checkpoint.")
        except Exception as e:
            logging.error(f"Error loading Motion Feature Model: {e}")
            raise

    @staticmethod
    def compute_polygon_area(points):
        """
        Compute area of a polygon using the shoelace formula
        """
        if len(points) < 3:
            return 0.0
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @staticmethod
    def compute_polygon_perimeter(points):
        """
        Compute perimeter of a polygon
        """
        if len(points) < 2:
            return 0.0
        return np.sum(np.sqrt(np.sum((points - np.roll(points, 1, axis=0))**2, axis=1)))

    def extract_motion_features(self, tracking_data: dict, inference_folder: str):
        """
        Extracts motion features from tracking data.
        Processes them through the MotionFeatureNet model.
        Saves motion features to 'motion_features.json' and 'motion_features.csv'.
        Outputs 'detection_summary.json' with total detections and detections per class.
        """
        logging.info("Extracting motion features from tracking data...")
        if not tracking_data:
            logging.warning("No tracking data available for motion feature extraction.")
            return

        features = {}
        detections_per_class = {
            "M1": 0,
            "M2": 0,
            "MDA": 0,
            "FB": 0
        }
        total_detections = 0

        for track_id, track in tqdm(tracking_data.items(), desc="Processing tracks"):
            if len(track) < 3:  # Need at least 3 points for acceleration
                logging.warning(f"Track {track_id} has less than 3 points, skipping.")
                continue

            # Extract temporal sequences
            frames = np.array([t['frame'] for t in track])
            x_pos = np.array([t['x'] for t in track])
            y_pos = np.array([t['y'] for t in track])
            scores = np.array([t['score'] for t in track])

            # Time intervals (assuming constant frame rate)
            dt = np.diff(frames)
            if np.any(dt == 0):
                logging.warning(f"Zero time interval in track {track_id}, skipping.")
                continue

            # Compute velocities (already in pixels)
            dx = np.diff(x_pos)
            dy = np.diff(y_pos)
            vx = dx / dt
            vy = dy / dt

            # Compute speeds and directions
            speeds = np.sqrt(vx**2 + vy**2)
            directions = np.arctan2(vy, vx)

            # Compute accelerations
            if len(dt) > 1:
                ax = np.diff(vx) / dt[:-1]
                ay = np.diff(vy) / dt[:-1]
                accelerations = np.sqrt(ax**2 + ay**2)
            else:
                accelerations = np.array([0.0])

            # Compute path characteristics
            total_distance = np.sum(np.sqrt(dx**2 + dy**2))
            displacement = np.sqrt((x_pos[-1] - x_pos[0])**2 + (y_pos[-1] - y_pos[0])**2)
            directness = displacement / total_distance if total_distance > 0 else 0

            # Compute shape changes
            areas = []
            perimeters = []
            for point in track:
                # Convert segmentation to array of points
                if isinstance(point['segmentation'], list) and len(point['segmentation']) > 0:
                    # Assuming 'segmentation' is a list of points
                    poly = np.array(point['segmentation']).reshape(-1, 2)
                    areas.append(self.compute_polygon_area(poly))
                    perimeters.append(self.compute_polygon_perimeter(poly))

            if areas:  # Only compute if we have valid areas
                area_changes = np.diff(areas) / dt if len(areas) > 1 else np.array([0.0])
                perimeter_changes = np.diff(perimeters) / dt if len(perimeters) > 1 else np.array([0.0])
            else:
                area_changes = np.array([0.0])
                perimeter_changes = np.array([0.0])

            # Prepare input features (15 features)
            feature_vector = [
                float(np.mean(speeds)),                      # mean_speed
                float(np.max(speeds)),                       # max_speed
                float(np.mean(accelerations)),               # mean_acceleration
                float(np.mean(directions)),                  # mean_direction
                float(np.std(directions)),                   # direction_change_rate
                float(directness),                           # path_directness
                float(total_distance),                        # total_distance
                float(frames[-1] - frames[0]),               # track_duration
                float(np.mean(areas)) if areas else 0.0,      # mean_area
                float(np.mean(area_changes)),                 # area_change_rate
                float(np.mean(perimeters)) if perimeters else 0.0,  # mean_perimeter
                float(np.mean(perimeter_changes)),            # perimeter_change_rate
                float(np.mean(scores)),                       # mean_detection_score
                float(np.min(scores)),                        # min_detection_score
                float(np.std(scores))                         # detection_score_std
            ]

            # Scale features using the loaded scaler
            scaled_features = self.scaler.transform([feature_vector])  # Shape: [1,15]

            # Convert feature vector to tensor
            input_features = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)  # Shape: [1,15]

            # Pass through the model to get predictions
            with torch.no_grad():
                model_output = self.motion_model(input_features)
                # Get predicted class
                predicted_class = torch.argmax(model_output, dim=1).item()
                # Optionally, get probabilities
                probabilities = torch.softmax(model_output, dim=1).cpu().numpy().tolist()[0]

            # Store motion features along with model predictions
            features[track_id] = {
                'mean_speed': feature_vector[0],
                'max_speed': feature_vector[1],
                'mean_acceleration': feature_vector[2],
                'mean_direction': feature_vector[3],
                'direction_change_rate': feature_vector[4],
                'path_directness': feature_vector[5],
                'total_distance': feature_vector[6],
                'track_duration': feature_vector[7],
                'mean_area': feature_vector[8],
                'area_change_rate': feature_vector[9],
                'mean_perimeter': feature_vector[10],
                'perimeter_change_rate': feature_vector[11],
                'mean_detection_score': feature_vector[12],
                'min_detection_score': feature_vector[13],
                'detection_score_std': feature_vector[14],
                'predicted_class': predicted_class,
                'probabilities': probabilities  # Uncomment if needed
            }

            # Update detections_per_class and total_detections
            class_label = self.get_class_label(predicted_class)
            if class_label in detections_per_class:
                detections_per_class[class_label] += 1
            else:
                detections_per_class[class_label] = 1
            total_detections += 1

        if not features:
            logging.warning("No valid motion features computed.")
            return

        # Convert features to DataFrame
        features_df = pd.DataFrame.from_dict(features, orient='index')
        logging.info(f"Computed motion features for {len(features_df)} tracks.")

        # Save motion features to CSV
        motion_features_file = os.path.join(inference_folder, "motion_features.csv")
        try:
            features_df.to_csv(motion_features_file, index_label='track_id')
            logging.info(f"Motion features saved to {motion_features_file}")
        except Exception as e:
            logging.error(f"Error saving motion features CSV: {e}")
            raise

        # Save features in a JSON file as well
        motion_features_json = os.path.join(inference_folder, "motion_features.json")
        try:
            features_df.to_json(motion_features_json, orient='index')
            logging.info(f"Motion features saved to {motion_features_json}")
        except Exception as e:
            logging.error(f"Error saving motion features JSON: {e}")
            raise

        # Save detection summary
        detection_summary = {
            "total_detections": total_detections,
            "detections_per_class": detections_per_class
        }

        detection_summary_file = os.path.join(inference_folder, "detection_summary.json")
        try:
            with open(detection_summary_file, 'w') as f:
                json.dump(detection_summary, f, indent=4)
            logging.info(f"Detection summary saved to {detection_summary_file}")
        except Exception as e:
            logging.error(f"Error saving detection summary JSON: {e}")
            raise

    def get_class_label(self, class_id: int) -> str:
        """
        Maps class_id to class_label using predefined mapping.
        """
        class_mapping = {
            0: "MDA",
            1: "FB",
            2: "M1",
            3: "M2"
        }
        return class_mapping.get(class_id, "unknown")

def main():
    """
    Main function to execute Phase 2: Motion Feature Extraction.
    """
    # Hardcoded paths and parameters
    BASE_DIR = "/scratch/rl4789/CellClassif"
    motion_model_config_path = "/scratch/rl4789/CellClassif/configs/motion_configs/motion_model_config.json"
    cell_type = "M2"
    video_id = "m2-013"
    output_dir = "/scratch/rl4789/CellClassif/inference_results/m2-013_inference_20241230_053002"
    tracking_data_path = "/scratch/rl4789/CellClassif/inference_results/m2-013_inference_20241230_053002/tracking_data.json"

    # Load tracking data
    if not os.path.exists(tracking_data_path):
        logging.error(f"Tracking data file {tracking_data_path} does not exist.")
        exit(1)
    with open(tracking_data_path, 'r') as f:
        tracking_data = json.load(f)
    logging.info(f"Loaded tracking data from {tracking_data_path}")

    # Initialize Phase2_MotionFeatureExtractor
    phase2 = Phase2_MotionFeatureExtractor(
        motion_model_config_path=motion_model_config_path,
        output_dir=output_dir,
        cell_type=cell_type,
        video_id=video_id
    )

    # Extract motion features and output detection summary
    phase2.extract_motion_features(tracking_data, output_dir)

    logging.info("Phase 2 processing completed successfully.")

if __name__ == "__main__":
    main()
