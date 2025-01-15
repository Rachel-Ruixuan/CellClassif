# inference_pipeline.py

import argparse
import os
import json
from datetime import datetime
import logging

from phase1 import Phase1_FrameProcessor
from phase2 import Phase2_MotionFeatureExtractor
from phase3 import Phase3_EnsemblerAndVisualizer
from config import TRAINING_CONFIG  # Ensure this path is correct

# Configure logging for the main script
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler("inference_pipeline.log")  # Logs to a file
    ]
)

BASE_DIR = "/scratch/rl4789/CellClassif"

def main():
    """
    Main function to execute the inference pipeline phases based on command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Inference Pipeline with Three Phases")
    parser.add_argument('--run_phase1', action='store_true', help='Run Phase 1: Frame Processing')
    parser.add_argument('--run_phase2', action='store_true', help='Run Phase 2: Motion Feature Extraction')
    parser.add_argument('--run_phase3', action='store_true', help='Run Phase 3: Ensembling & Visualization')
    parser.add_argument('--run_all', action='store_true', help='Run all phases sequentially')
    args = parser.parse_args()

    # Define paths and parameters
    mixed_detectron_path = "/scratch/rl4789/CellClassif/src/segmentation/models/mixed_detectron/model_final.pth"  # Update as needed
    motion_model_config_path = "/scratch/rl4789/CellClassif/configs/motion_configs/motion_model_config.json"  # Updated path
    output_dir = "/scratch/rl4789/CellClassif/inference_results"
    cell_type = "MDA"  # Assuming single category; adjust if multiple categories per video
    video_id = "100MDA_4034"
    image_dir = os.path.join(BASE_DIR, "data", "raw", cell_type, video_id)

    # Create a unique inference folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_folder = os.path.join(output_dir, f"{video_id}_inference_{timestamp}")
    os.makedirs(inference_folder, exist_ok=True)

    # Define class mapping (ensure consistency across phases)
    class_mapping = {
        0: "MDA",
        1: "FB",
        2: "M1",
        3: "M2"
    }

    # Set input_dim and num_classes based on training setup
    input_dim = 15  # As per training
    num_classes = 4  # As per training

    # Load motion model configuration (assuming it's a JSON file)
    try:
        with open(motion_model_config_path, 'r') as f:
            motion_model_config = json.load(f)
        logging.info(f"Loaded motion model configuration from {motion_model_config_path}")
    except Exception as e:
        logging.error(f"Error loading motion model configuration: {e}")
        exit(1)

    # Initialize phase instances
    phase1 = Phase1_FrameProcessor(
        mixed_detectron_path=mixed_detectron_path,
        output_dir=output_dir,
        cell_type=cell_type,
        video_id=video_id
    )

    phase2 = Phase2_MotionFeatureExtractor(
        motion_model_config_path=motion_model_config_path,  # Updated to path
        output_dir=output_dir,
        cell_type=cell_type,
        video_id=video_id
    )

    phase3 = Phase3_EnsemblerAndVisualizer(
        class_mapping=class_mapping,
        output_dir=output_dir,
        video_id=video_id,
        inference_folder=inference_folder
    )

    # Determine which phases to run
    if args.run_all:
        # Run Phase 1
        logging.info("\n=== Running Phase 1: Frame Processing ===")
        phase1.process_video_frames(image_dir, inference_folder)

        # Run Phase 2
        logging.info("\n=== Running Phase 2: Motion Feature Extraction ===")
        # Load tracking data from Phase 1
        tracking_output_file = os.path.join(inference_folder, "tracking_data.json")
        if not os.path.exists(tracking_output_file):
            logging.error(f"Tracking data file {tracking_output_file} does not exist. Please run Phase 1 first.")
            exit(1)
        with open(tracking_output_file, 'r') as f:
            tracking_data = json.load(f)
        phase2.extract_motion_features(tracking_data, inference_folder)

        # Run Phase 3
        logging.info("\n=== Running Phase 3: Ensembling & Visualization ===")
        # Load motion features and segmentation results
        motion_features = phase3.load_motion_predictions()
        segmentation_results = phase3.load_segmentation_results()
        final_predictions = phase3.perform_ensembling(segmentation_results, motion_features)
        phase3.annotate_frames(final_predictions, image_dir)
        final_video_path = os.path.join(output_dir, f"{video_id}_prediction_{timestamp}.mp4")
        annotated_frames_dir = os.path.join(inference_folder, "annotated_frames")
        phase3.create_final_video(annotated_frames_dir, final_video_path)
    else:
        # Run individual phases based on arguments
        if args.run_phase1:
            logging.info("\n=== Running Phase 1: Frame Processing ===")
            phase1.process_video_frames(image_dir, inference_folder)

        if args.run_phase2:
            logging.info("\n=== Running Phase 2: Motion Feature Extraction ===")
            # Load tracking data from Phase 1
            tracking_output_file = os.path.join(inference_folder, "tracking_data.json")
            if not os.path.exists(tracking_output_file):
                logging.error(f"Tracking data file {tracking_output_file} does not exist. Please run Phase 1 first.")
                exit(1)
            with open(tracking_output_file, 'r') as f:
                tracking_data = json.load(f)
            phase2.extract_motion_features(tracking_data, inference_folder)

        if args.run_phase3:
            logging.info("\n=== Running Phase 3: Ensembling & Visualization ===")
            # Load motion features and segmentation results
            motion_features = phase3.load_motion_predictions()
            segmentation_results = phase3.load_segmentation_results()
            final_predictions = phase3.perform_ensembling(segmentation_results, motion_features)
            phase3.annotate_frames(final_predictions, image_dir)
            final_video_path = os.path.join(output_dir, f"{video_id}_prediction_{timestamp}.mp4")
            annotated_frames_dir = os.path.join(inference_folder, "annotated_frames")
            phase3.create_final_video(annotated_frames_dir, final_video_path)

        # If no arguments are provided, inform the user
        if not any([args.run_phase1, args.run_phase2, args.run_phase3]):
            print("No phases specified to run. Use --run_phase1, --run_phase2, --run_phase3, or --run_all.")

if __name__ == "__main__":
    main()
