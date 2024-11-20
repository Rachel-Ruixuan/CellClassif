import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort

class CellTracker:
    def __init__(self, prediction_dir, raw_dir, output_dir):
        self.prediction_dir = prediction_dir
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.tracker_params = {
            'max_age': 30,
            'n_init': 3,
            'max_iou_distance': 0.7,
            'max_cosine_distance': 0.3,
            'nn_budget': 100
        }
    
    def process_video(self, video_id):
        # Create a new tracker instance for each video
        tracker = DeepSort(**self.tracker_params)
        
        # Load predictions
        pred_path = os.path.join(self.prediction_dir, video_id, 'predictions.json')
        with open(pred_path, 'r') as f:
            predictions = json.load(f)
        
        tracked_predictions = {}
        next_track_id = 1  # Initialize track ID counter
        
        # Sort frames by time
        frame_names = sorted(
            predictions.keys(),
            key=lambda x: int(x.split('_t')[1].split('_')[0])
        )
        
        # Track mapping to ensure consistent IDs
        id_mapping = {}
        
        for frame_name in tqdm(frame_names, desc=f"Tracking {video_id}"):
            frame_path = os.path.join(self.raw_dir, video_id, frame_name)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Could not read frame: {frame_path}")
                continue
            
            frame_preds = predictions[frame_name]
            detections = []
            original_preds = []
            
            for pred in frame_preds:
                bbox = pred['bbox']
                detections.append([[bbox[0], bbox[1], bbox[2], bbox[3]], pred['score']])
                original_preds.append(pred)
            
            tracks = tracker.update_tracks(detections, frame=frame)
            
            frame_annotations = []
            for track, original_pred in zip(tracks, original_preds):
                if not track.is_confirmed():
                    continue
                
                # Map DeepSort track ID to our sequential ID
                if track.track_id not in id_mapping:
                    id_mapping[track.track_id] = next_track_id
                    next_track_id += 1
                
                annotation = original_pred.copy()
                annotation['track_id'] = id_mapping[track.track_id]
                frame_annotations.append(annotation)
            
            tracked_predictions[frame_name] = frame_annotations
        
        # Save results
        os.makedirs(os.path.join(self.output_dir, video_id), exist_ok=True)
        output_path = os.path.join(self.output_dir, video_id, 'track_id.json')
        with open(output_path, 'w') as f:
            json.dump(tracked_predictions, f, indent=2)
        
        return tracked_predictions

def main():
    base_dir = "/scratch/rl4789/CellClassif"
    cell_types = ["M1", "M2"]
    
    for cell_type in cell_types:
        print(f"\nProcessing {cell_type} videos...")
        
        prediction_dir = os.path.join(base_dir, "data/interim/segmentation", cell_type, "predictions")
        raw_dir = os.path.join(base_dir, "data/raw", cell_type)
        output_dir = os.path.join(base_dir, "data/interim/tracking", cell_type)
        
        tracker_instance = CellTracker(prediction_dir, raw_dir, output_dir)
        
        for video_id in os.listdir(prediction_dir):
            if os.path.isdir(os.path.join(prediction_dir, video_id)):
                tracker_instance.process_video(video_id)

if __name__ == "__main__":
    main()