# Cell Classification Project
This project performs segmentation, tracking, and classification on different types of cells using Detectron2 and DeepSort. The primary focus is on inference using a custom-trained mixed Detectron2 model to achieve high-confidence predictions.

## Table of Contents
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data Organization](#data-organization)
- [Current Workflow](#current-workflow)
  - [Training (Segmentation only)](#training-segmentation-only)
  - [Inference (Detectron2 only)](#inference-detectron2-only)
- [Expected Complete Workflow](#expected-complete-workflow)
  - [Training](#training-1)
    - [Segmentation Model](#segmentation-model)
    - [Motion Model](#motion-model)
  - [Inference](#inference)
    - [Phase 1: Detectron2 Inference](#phase-1-detectron2-inference)
    - [Phase 2: Motion Feature Extraction](#phase-2-motion-feature-extraction)
    - [Phase 3: Ensemble Classification](#phase-3-ensemble-classification)
- [Results](#results)
- [Progress Overview](#progress-overview)
  - [Current Status](#current-status)
  - [Future Work](#future-work)
- [Important Notes](#important-notes)


## Project Structure
```bash
CellClassif/
├── configs/
│   ├── detectron_configs/
│   └── motion_configs/
├── data/
│   ├── interim/
│   │   ├── segmentation/
│   │   │   ├── FB/
│   │   │   │   ├── predictions/
│   │   │   │   │   ├── 100FB_001/
│   │   │   │   │   ├── 100FB_002/
│   │   │   │   │   └── ...
│   │   │   ├── {cell_type}/
│   │   └── tracking/
│   │       ├── FB/
│   │       │   ├── 100FB_001/
│   │       │   ├── 100FB_002/
│   │       │   └── ...
│   │       ├── {cell_type}/
│   ├── processed/
│   │   ├── detectron_format/
│   │   │   ├── {cell_type1}/
│   │   │   ├── {cell_type2}/
│   │   │   ├── ...
│   │   │   └── mixed_detectron/
│   │   └── motion/
│   │       ├── {cell_type}/
│   └── raw/
│       ├── FB/
│       │   ├── 100FB_001/
│       │   └── ...
│       ├── {cell_type}/
├── inference_results/
│   └── {cell_type}_{video_id}_inference_{datetime}/
│       └── visualizations/
├── notebooks/
├── src/
│   ├── segmentation/
│   │   ├── models/
│   │   │   ├── {cell_type1}/
│   │   │   ├── {cell_type2}/
│   │   │   ├── ...
│   │   │   └── mixed_detectron/
│   │   ├── predict_detectron.py
│   │   ├── train_detectron.py
│   │   └── train_mixed_detectron.py
│   ├── inference/
│   │   ├── config.py
│   │   ├── inference_pipeline.py
│   │   ├── phase1.py
│   │   ├── phase1_integrated.py
│   │   ├── phase2.py
│   │   └── phase3.py
│   ├── motion/
│   │   └── results/
│   │   └── train_motion_model.py
│   └── tracking/
│       ├── analyze_tracks.py
│       └── assign_trackid.py
└── README.md
```
## Setup
### Clone the Repository:
```bash
git clone https://github.com/Rachel-Ruixuan/CellClassif.git
cd CellClassif
```
### Create a Virtual Environment and Install Dependencies:

```bash
conda create -n {env_name} python=3.8
conda activate {env_name} 
pip install -r requirements.txt
```

### Configure Cell Types:

Before running any scripts, ensure that cell types are correctly specified in the configuration files.

- For Segmentation (`src/segmentation/train_detectron.py` and `src/segmentation/predict_detectron.py`):

```python
cell_types = ["MDA", "FB", "M1", "M2", "ADSC"]  # Modify with your cell types
```

- For Tracking (`src/tracking/assign_trackid.py`):
```python
cell_types = ["MDA", "FB", "M1", "M2", "ADSC"]  # Modify with your cell types
```

- For Motion Features (`src/motion/train_motion_model.py`):

```python
CONFIG = {
    'datasets': ['MDA', 'FB', 'M1', 'M2', 'ADSC'],  # Modify with your cell types
    'data_config': {
        'MDA': {  # Add/modify cell types
            'feature_file': '/path/to/MDA/all_features.csv',
            'class_name': 'MDA'
        },
        'FB': {   # Add/modify cell types
            'feature_file': '/path/to/FB/all_features.csv',
            'class_name': 'FB'
        },
        # Add other cell types as needed
    },
    ...
}
```

## Data Organization
#### Raw Data:

- Place raw images in the `data/raw/<CELL_TYPE>/` directory.
- Ensure correct naming format or similar to the format: `<CELL_TYPE>_video_id/100<CELL_TYPE>_video_id_tXX_ch00.jpg`

**Example**:
```bash
data/raw/MDA/100MDA_4034/
    └── 100MDA_4034_t01_ch00.jpg
    └── 100MDA_4034_t02_ch00.jpg
    └── ...
```

## Current Workflow
The project is divided into several phases, each responsible for a specific task in the cell classification workflow. Currently, the focus is on Phase 1: Inference using the custom-trained mixed Detectron2 model.

### Training (Segmentation only)
#### Train Separate Detectron2 Model (for each cell type):

```bash
python src/segmentation/train_detectron.py
```
**Purpose**: Train the mixed Detectron2 model on your dataset to perform instance segmentation.

#### Run Inference for Larger Training Dataset:

```bash
python src/segmentation/predict_detectron.py
```

**Purpose**: Use the specifically trained Detectron2 models on each cell type to perform inference on raw images that particular cell type to generate segmentation masks, significantly enlarging the dataset for training the final `mixed_detectron`.

#### Train Final Detectron2 Model (for classifying between all cell types):

```bash
python src/segmentation/train_mixed_detectron.py
```

### Inference (Detectron2 only)

```bash
python src/inference/phase1_integrated.py
```
**Purpose**: Perform detection and tracking on raw images, save segmentation and tracking results, visualize tracking by overlaying bounding boxes and masks, and compile annotated frames into a video.

**Notes**:
- Paths are hardcoded in `phase1_integrated.py` based on your directory structure.
- Ensure that the video_id and corresponding directories are correctly specified within the script.
- The script outputs an annotated video in the `inference_results/<VIDEO_ID>_inference_<TIMESTAMP>/` directory.

## Expected Complete Workflow
### Training
-  #### Segmentation Model
    - Go through the same steps as specified in [Training (Segmentation only)](#training-segmentation-only)
- #### Motion Model
  - Tracking:
    - Assign Track IDs:
      ```bash
      python src/tracking/assign_trackid.py
      ```
      **Purpose**: Assign unique track_ids to detected cells across frames using DeepSort.

    - Analyze Tracks:
      ```bash
      python src/tracking/analyze_tracks.py
      ```
      **Purpose**: Analyze tracking data, extract features, and prepare for further motion analysis.

  - Motion Analysis - Train Motion Model: 
    ```bash
    python src/motion/train_motion_model.py
    ```
    **Purpose**: Train a motion model to analyze the movement patterns of tracked cells, enhancing classification accuracy.

### Inference
The inference process consists of three phases:
#### Phase 1: Detectron2 Inference
Go thourgh the steps specified in [Inference (Detectron2 only)](#inference-detectron2-only)

#### Phase 2: Motion Feature Extraction
Motion features are extracted by analyzing segmentations from Phase 1.
```bash
python src/inference/phase2_integrated.py
```

#### Phase 3: Ensemble Classification
Ensemble predictions from Detectron2 and the motion model are combined to improve classification accuracy.

```bash
python src/inference/phase3_integrated.py
```
**Note**: It is also advisible to merge the three phases into one `inference_pipeline.py` by calling the three phases altogether.


## Results
![Inference Video](https://raw.githubusercontent.com/Rachel-Ruixuan/CellClassif/main/inference_results/100MDA_4034_inference_20241230_020032/Screenshot.png)

Watch the [inference video](https://raw.githubusercontent.com/Rachel-Ruixuan/CellClassif/main/inference_results/100MDA_4034_inference_20241230_020032/100MDA_4034_annotated_20241230_020214.mp4) to see the model in action.


## Progress Overview
- ### Current Status
  - **Model Training:** Trained Detectron2 models using mixed_detectron for classification.
  - **Inference:** Generated a final video demonstrating prediction results.
  - **Challenges:** 
    - Integration of tracking data did not yield expected improvements due to issues accessing probits output and limitations of the current motion model.

- ### Future Work
  - Explore advanced motion models to enhance ensemble methods.
  - Refine the inference pipeline based on evaluation metrics.


## Important Notes
- Image Sizes Differ by Cell Type:
    - MDA: (2048, 2048)
    - FB/ADSC: (1224, 904)

Ensure that image size configurations in relevant scripts are correctly set to handle varying dimensions.

- Confidence Scores:
    - The custom-trained mixed Detectron2 model achieves high-confidence predictions (≥70%).
    - Confidence scores are crucial for reliable tracking and classification.


#### High Confidence Scores:
The mixed Detectron2 model demonstrates high-confidence predictions, ensuring reliable detection and tracking without the need for an ensemble approach at this stage.








