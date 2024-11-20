import os
import json
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import re

class CellTrackingAnalysis:
    def __init__(self, video_dir, cell_type):
        """
        Initialize the analyzer with the video directory containing track_id.json
        """
        self.video_dir = video_dir
        self.track_id_path = os.path.join(video_dir, 'track_id.json')
        self.cell_type = cell_type
        self.frame_data = None
        self.track_features = {}
        # Set image size based on cell type
        self.image_size = (1224, 904) if cell_type in ["FB", "ADSC"] else (2048, 2048)
        print(f"Using image size {self.image_size} for {cell_type}")
        
    def load_tracking_data(self):
        """
        Load and organize tracking data from track_id.json
        """
        if not os.path.exists(self.track_id_path):
            raise FileNotFoundError(f"track_id.json not found in {self.video_dir}")
            
        with open(self.track_id_path, 'r') as f:
            self.frame_data = json.load(f)
            
        if not self.frame_data:
            raise ValueError(f"Empty tracking data in {self.track_id_path}")
            
        # Convert frame numbers for temporal analysis
        self.frame_numbers = {}
        for frame_name in self.frame_data.keys():
            match = re.search(r't(\d+)_', frame_name)
            if match:
                self.frame_numbers[frame_name] = int(match.group(1))
            else:
                print(f"Warning: Could not extract frame number from {frame_name}")
    
    def extract_track_trajectories(self):
        """
        Extract trajectories for each tracked cell
        """
        trajectories = {}
        
        for frame_name, annotations in self.frame_data.items():
            if frame_name not in self.frame_numbers:
                continue
                
            frame_num = self.frame_numbers[frame_name]
            
            for ann in annotations:
                try:
                    track_id = ann['track_id']
                    if track_id not in trajectories:
                        trajectories[track_id] = []
                    
                    # Get center point from bbox
                    bbox = ann['bbox']
                    center_x = (bbox[0] + bbox[2]) / 2  # Already in pixel coordinates
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    # Store frame number and position
                    trajectories[track_id].append({
                        'frame': frame_num,
                        'x': center_x,
                        'y': center_y,
                        'bbox': bbox,
                        'segmentation': ann['segmentation'],
                        'score': ann['score']  # Add detection score
                    })
                except KeyError as e:
                    print(f"Warning: Missing key {e} in annotation: {ann}")
                    continue
        
        if not trajectories:
            raise ValueError("No valid trajectories extracted from tracking data")
            
        # Sort each trajectory by frame number
        for track_id in trajectories:
            trajectories[track_id].sort(key=lambda x: x['frame'])
            
        return trajectories
    
    def compute_motion_features(self, trajectories):
        """
        Compute motion-based features for each track
        """
        features = {}
        
        for track_id, track in trajectories.items():
            if len(track) < 3:  # Need at least 3 points for acceleration
                print(f"Warning: Track {track_id} has less than 3 points, skipping")
                continue
            
            try:
                # Extract temporal sequences
                frames = np.array([t['frame'] for t in track])
                x_pos = np.array([t['x'] for t in track])
                y_pos = np.array([t['y'] for t in track])
                scores = np.array([t['score'] for t in track])
                
                # Time intervals (assuming constant frame rate)
                dt = np.diff(frames)
                if np.any(dt == 0):
                    print(f"Warning: Zero time interval in track {track_id}, skipping")
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
                        poly = np.array(point['segmentation'][0]).reshape(-1, 2)  # Note the [0] index
                        areas.append(self.compute_polygon_area(poly))
                        perimeters.append(self.compute_polygon_perimeter(poly))
                
                if areas:  # Only compute if we have valid areas
                    area_changes = np.diff(areas) / dt if len(areas) > 1 else np.array([0.0])
                    perimeter_changes = np.diff(perimeters) / dt if len(perimeters) > 1 else np.array([0.0])
                else:
                    area_changes = np.array([0.0])
                    perimeter_changes = np.array([0.0])
                
                # Store computed features
                features[track_id] = {
                    'mean_speed': float(np.mean(speeds)),
                    'max_speed': float(np.max(speeds)),
                    'mean_acceleration': float(np.mean(accelerations)),
                    'mean_direction': float(np.mean(directions)),
                    'direction_change_rate': float(np.std(directions)),
                    'path_directness': float(directness),
                    'total_distance': float(total_distance),
                    'track_duration': float(frames[-1] - frames[0]),
                    'mean_area': float(np.mean(areas)) if areas else 0.0,
                    'area_change_rate': float(np.mean(area_changes)),
                    'mean_perimeter': float(np.mean(perimeters)) if perimeters else 0.0,
                    'perimeter_change_rate': float(np.mean(perimeter_changes)),
                    'mean_detection_score': float(np.mean(scores)),
                    'min_detection_score': float(np.min(scores)),
                    'detection_score_std': float(np.std(scores)),
                    'category': self.cell_type
                }
                
            except Exception as e:
                print(f"Error processing track {track_id}: {str(e)}")
                continue
        
        if not features:
            raise ValueError("No valid features computed from trajectories")
            
        return features
    
    @staticmethod
    def compute_polygon_area(points):
        """
        Compute area of a polygon using the shoelace formula
        """
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    @staticmethod
    def compute_polygon_perimeter(points):
        """
        Compute perimeter of a polygon
        """
        return np.sum(np.sqrt(np.sum((points - np.roll(points, 1, axis=0))**2, axis=1)))
    
    def extract_all_features(self):
        """
        Extract all features and prepare for model training
        """
        try:
            print("Loading tracking data...")
            self.load_tracking_data()
            
            print("Extracting trajectories...")
            trajectories = self.extract_track_trajectories()
            print(f"Found {len(trajectories)} trajectories")
            
            print("Computing motion features...")
            features = self.compute_motion_features(trajectories)
            print(f"Computed features for {len(features)} tracks")
            
            if not features:
                raise ValueError("No features computed")
            
            # Convert to pandas DataFrame
            df = pd.DataFrame.from_dict(features, orient='index')
            print(f"Feature columns: {df.columns.tolist()}")
            
            # Scale features
            feature_columns = [col for col in df.columns if col != 'category']
            scaler = StandardScaler()
            df[feature_columns] = scaler.fit_transform(df[feature_columns])
            
            return df
            
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            raise

import os
import json
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import re

class CellTrackingAnalysis:
    def __init__(self, video_dir, cell_type):
        """
        Initialize the analyzer with the video directory containing track_id.json
        """
        self.video_dir = video_dir
        self.track_id_path = os.path.join(video_dir, 'track_id.json')
        self.cell_type = cell_type
        self.frame_data = None
        self.track_features = {}
        # Set image size based on cell type
        self.image_size = (1224, 904) if cell_type in ["FB", "ADSC"] else (2048, 2048)
        print(f"Using image size {self.image_size} for {cell_type}")
        
    def load_tracking_data(self):
        """
        Load and organize tracking data from track_id.json
        """
        if not os.path.exists(self.track_id_path):
            raise FileNotFoundError(f"track_id.json not found in {self.video_dir}")
            
        with open(self.track_id_path, 'r') as f:
            self.frame_data = json.load(f)
            
        if not self.frame_data:
            raise ValueError(f"Empty tracking data in {self.track_id_path}")
            
        # Convert frame numbers for temporal analysis
        self.frame_numbers = {}
        for frame_name in self.frame_data.keys():
            match = re.search(r't(\d+)_', frame_name)
            if match:
                self.frame_numbers[frame_name] = int(match.group(1))
            else:
                print(f"Warning: Could not extract frame number from {frame_name}")
    
    def extract_track_trajectories(self):
        """
        Extract trajectories for each tracked cell
        """
        trajectories = {}
        
        for frame_name, annotations in self.frame_data.items():
            if frame_name not in self.frame_numbers:
                continue
                
            frame_num = self.frame_numbers[frame_name]
            
            for ann in annotations:
                try:
                    track_id = ann['track_id']
                    if track_id not in trajectories:
                        trajectories[track_id] = []
                    
                    # Get center point from bbox
                    bbox = ann['bbox']
                    center_x = (bbox[0] + bbox[2]) / 2  # Already in pixel coordinates
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    # Store frame number and position
                    trajectories[track_id].append({
                        'frame': frame_num,
                        'x': center_x,
                        'y': center_y,
                        'bbox': bbox,
                        'segmentation': ann['segmentation'],
                        'score': ann['score']  # Add detection score
                    })
                except KeyError as e:
                    print(f"Warning: Missing key {e} in annotation: {ann}")
                    continue
        
        if not trajectories:
            raise ValueError("No valid trajectories extracted from tracking data")
            
        # Sort each trajectory by frame number
        for track_id in trajectories:
            trajectories[track_id].sort(key=lambda x: x['frame'])
            
        return trajectories
    
    def compute_motion_features(self, trajectories):
        """
        Compute motion-based features for each track
        """
        features = {}
        
        for track_id, track in trajectories.items():
            if len(track) < 3:  # Need at least 3 points for acceleration
                print(f"Warning: Track {track_id} has less than 3 points, skipping")
                continue
            
            try:
                # Extract temporal sequences
                frames = np.array([t['frame'] for t in track])
                x_pos = np.array([t['x'] for t in track])
                y_pos = np.array([t['y'] for t in track])
                scores = np.array([t['score'] for t in track])
                
                # Time intervals (assuming constant frame rate)
                dt = np.diff(frames)
                if np.any(dt == 0):
                    print(f"Warning: Zero time interval in track {track_id}, skipping")
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
                        poly = np.array(point['segmentation'][0]).reshape(-1, 2)  # Note the [0] index
                        areas.append(self.compute_polygon_area(poly))
                        perimeters.append(self.compute_polygon_perimeter(poly))
                
                if areas:  # Only compute if we have valid areas
                    area_changes = np.diff(areas) / dt if len(areas) > 1 else np.array([0.0])
                    perimeter_changes = np.diff(perimeters) / dt if len(perimeters) > 1 else np.array([0.0])
                else:
                    area_changes = np.array([0.0])
                    perimeter_changes = np.array([0.0])
                
                # Store computed features
                features[track_id] = {
                    'mean_speed': float(np.mean(speeds)),
                    'max_speed': float(np.max(speeds)),
                    'mean_acceleration': float(np.mean(accelerations)),
                    'mean_direction': float(np.mean(directions)),
                    'direction_change_rate': float(np.std(directions)),
                    'path_directness': float(directness),
                    'total_distance': float(total_distance),
                    'track_duration': float(frames[-1] - frames[0]),
                    'mean_area': float(np.mean(areas)) if areas else 0.0,
                    'area_change_rate': float(np.mean(area_changes)),
                    'mean_perimeter': float(np.mean(perimeters)) if perimeters else 0.0,
                    'perimeter_change_rate': float(np.mean(perimeter_changes)),
                    'mean_detection_score': float(np.mean(scores)),
                    'min_detection_score': float(np.min(scores)),
                    'detection_score_std': float(np.std(scores)),
                    'category': self.cell_type
                }
                
            except Exception as e:
                print(f"Error processing track {track_id}: {str(e)}")
                continue
        
        if not features:
            raise ValueError("No valid features computed from trajectories")
            
        return features
    
    @staticmethod
    def compute_polygon_area(points):
        """
        Compute area of a polygon using the shoelace formula
        """
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    @staticmethod
    def compute_polygon_perimeter(points):
        """
        Compute perimeter of a polygon
        """
        return np.sum(np.sqrt(np.sum((points - np.roll(points, 1, axis=0))**2, axis=1)))
    
    def extract_all_features(self):
        """
        Extract all features and prepare for model training
        """
        try:
            print("Loading tracking data...")
            self.load_tracking_data()
            
            print("Extracting trajectories...")
            trajectories = self.extract_track_trajectories()
            print(f"Found {len(trajectories)} trajectories")
            
            print("Computing motion features...")
            features = self.compute_motion_features(trajectories)
            print(f"Computed features for {len(features)} tracks")
            
            if not features:
                raise ValueError("No features computed")
            
            # Convert to pandas DataFrame
            df = pd.DataFrame.from_dict(features, orient='index')
            print(f"Feature columns: {df.columns.tolist()}")
            
            # Scale features
            feature_columns = [col for col in df.columns if col != 'category']
            scaler = StandardScaler()
            df[feature_columns] = scaler.fit_transform(df[feature_columns])
            
            return df
            
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            raise

def main():
    base_dir = "/scratch/rl4789/CellClassif"
    cell_types = ["M1", "M2"]
    
    for cell_type in cell_types:
        print(f"\nProcessing {cell_type} videos...")
        tracking_dir = os.path.join(base_dir, "data/interim/tracking", cell_type)
        
        all_features = []
        
        for video_id in os.listdir(tracking_dir):
            video_dir = os.path.join(tracking_dir, video_id)
            if not os.path.isdir(video_dir):
                continue
                
            try:
                print(f"\nProcessing video {video_id}")
                analyzer = CellTrackingAnalysis(video_dir, cell_type)
                features_df = analyzer.extract_all_features()
                
                # Add video ID column
                features_df['video_id'] = video_id
                all_features.append(features_df)
                
                # Save individual video features
                output_file = os.path.join(video_dir, 'cell_features.csv')
                features_df.to_csv(output_file)
                print(f"Features saved to {output_file}")
                
            except Exception as e:
                print(f"Error processing video {video_id}: {str(e)}")
                continue
        
        if all_features:
            # Combine all features for this cell type
            combined_features = pd.concat(all_features, ignore_index=True)
            
            # Save combined features
            output_dir = os.path.join(base_dir, "data/processed/classification", cell_type)
            os.makedirs(output_dir, exist_ok=True)
            combined_file = os.path.join(output_dir, 'all_features.csv')
            combined_features.to_csv(combined_file)
            print(f"\nCombined features for {cell_type} saved to {combined_file}")

if __name__ == "__main__":
    main()