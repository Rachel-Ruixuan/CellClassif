# Cell Classification Project

This project performs segmentation, tracking, and classification on different types of cells.

## Project Structure
``` bash
CellClassif/
├── data/
│   ├── raw/                   # Raw images
│   │   ├── MDA/               # Raw MDA images
│   │   │   └── 100MDA_video_id/
│   │   │       └── 100MDA_video_id_tXX_ch00.jpg
│   │   └── FB/                # Raw FB images
│   │       └── 100FB_video_id/
│   │           └── 100FB_video_id_tXX_ch00.jpg
│   ├── processed/             # Processed data
│   │   ├── detectron_format/  # Data formatted for Detectron2
│   │   │   ├── MDA/
│   │   │   │   ├── train/
│   │   │   │   │   ├── images/
│   │   │   │   │   └── annotations.json
│   │   │   │   └── val/
│   │   │   └── FB/
│   │   └── motion/    # Motion features
│   │       ├── MDA/
│   │       │   └── all_features.csv
│   │       └── FB/
│   └── interim/              # Intermediate results
│       ├── segmentation/     # Detectron2 predictions
│       │   ├── MDA/
│       │   │   └── predictions/
│       │   └── FB/
│       └── tracking/         # Tracking results
│           ├── MDA/
│           │   └── track_id.json
│           └── FB/
├── src/                      # Source code
├── configs/                  # Configuration files
└── notebooks/                # Jupyter notebooks
```

## Setup
#### 1. Clone the repository:
```bash
git clone https://github.com/Rachel-Ruixuan/CellClassif.git
cd CellClassif
```

#### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

#### 3. Configure cell types:
Before running any scripts, modify the cell types in the configuration sections:

- For segmentation (src/segmentation/train_detectron.py and predict_detectron.py):
```python
cell_types = ["MDA", "FB"]  # Modify with your cell types
```

- For tracking (src/tracking/assign_trackid.py):
```python
cell_types = ["MDA", "FB"]  # Modify with your cell types
```

- For motion features (src/motion/train.py):
```python
CONFIG = {
    'datasets': ['MDA', 'FB'],  # Modify with your cell types
    'data_config': {
        'MDA': {  # Add/modify cell types
            'feature_file': '/path/to/MDA/all_features.csv',
            'class_name': 'MDA'
        },
        'FB': {   # Add/modify cell types
            'feature_file': '/path/to/FB/all_features.csv',
            'class_name': 'FB'
        }
    },
    ...
}
```

## Pipeline Usage

#### 1. Data Organization:
   - Place raw images in appropriate directories under `data/raw/<CELL_TYPE>/`
   - Ensure correct naming format: `100<CELL_TYPE>_video_id/xxx.jpg`

#### 2. Segmentation:
```bash
# Train Detectron2 model
python src/segmentation/train_detectron.py

# Run inference
python src/segmentation/predict_detectron.py
```

#### 3. Tracking:
```bash
# Assign track IDs
python src/tracking/assign_trackid.py

# Extract tracking features
python src/tracking/analyze_tracks.py
```

#### 4. Motion:
```bash
# Train motion model
python src/motion/train.py
```

## Important Notes
- Image sizes may differ by cell type:
  - MDA: (2048, 2048)
  - FB/ADSC: (1224, 904)
- Ensure correct image size configuration in relevant scripts
- Check output directories for results and logs

## Status
🚧 This project is currently under active development 🚧