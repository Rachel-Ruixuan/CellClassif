# Cell Classification Project

## Project Structure
```
CellClassif/
├── data/               # Data directory (not tracked by git)
├── src/               # Source code
├── configs/           # Configuration files
└── notebooks/         # Jupyter notebooks
```

## Setup
1. Clone the repository:
```bash
git clone https://github.com/Rachel-Ruixuan/CellClassif.git
cd CellClassif
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Data organization:
- Place raw images in `data/raw/`
- Processed data will be saved in `data/processed/`
- Intermediate results in `data/interim/`

## Usage
1. Segmentation:
```bash
python src/segmentation/train_detectron.py
python src/segmentation/predict_detectron.py
```

2. Tracking:
```bash
python src/tracking/assign_trackid.py
python src/tracking/analyze_tracks.py
```

3. Classification:
```bash
python src/classification/train.py
```
