import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Configuration
CONFIG = {
    'datasets': ['MDA', 'FB', 'M1', 'M2'],
    'data_config': {
        'MDA': {
            'feature_file': '/scratch/rl4789/CellClassif/data/processed/motion/MDA/all_features.csv',
            'class_name': 'MDA'
        },
        'FB': {
            'feature_file': '/scratch/rl4789/CellClassif/data/processed/motion/FB/all_features.csv',
            'class_name': 'FB'
        },
        'M1': {
            'feature_file': '/scratch/rl4789/CellClassif/data/processed/motion/M1/all_features.csv',
            'class_name': 'M1'
        },
        'M2': {
            'feature_file': '/scratch/rl4789/CellClassif/data/processed/motion/M2/all_features.csv',
            'class_name': 'M2'
        }
    },
    'training': {
        'batch_size': 32,
        'num_epochs': 200,
        'learning_rate': 0.0005,
        'weight_decay': 0.02,
        'val_split': 0.2,
        'random_seed': 42,
        'scheduler_config': {
            'T_0': 50,
            'T_mult': 1,
            'eta_min': 1e-6
        }
    },
    'model': {
        'hidden_dims': [128, 64, 32],
        'dropout_rate': 0.5
    }
}

class MotionFeatureDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.config = config
        self.is_train = is_train
        print("Initializing MotionFeatureDataset...")
        
        # Load features from all cell types
        features_list = []
        for dataset_name in config['datasets']:
            try:
                df = pd.read_csv(config['data_config'][dataset_name]['feature_file'])
                
                # Handle index column if present
                if 'Unnamed: 0' in df.columns:
                    df = df.drop('Unnamed: 0', axis=1)
                
                # Add category if not present
                if 'category' not in df.columns:
                    df['category'] = config['data_config'][dataset_name]['class_name']
                
                print(f"Loaded features for {dataset_name}: {len(df)} rows")
                features_list.append(df)
                
            except Exception as e:
                print(f"Error loading features from {dataset_name}: {str(e)}")
                continue
        
        if not features_list:
            raise ValueError("No feature data loaded")
        
        # Combine all features
        self.features_df = pd.concat(features_list, ignore_index=True)
        
        # Drop any rows with NaN values
        initial_len = len(self.features_df)
        self.features_df = self.features_df.dropna()
        if len(self.features_df) < initial_len:
            print(f"Dropped {initial_len - len(self.features_df)} rows with NaN values")
        
        # Define feature columns (excluding category and video_id)
        self.feature_columns = [
            'mean_speed', 'max_speed', 'mean_acceleration', 'mean_direction',
            'direction_change_rate', 'path_directness', 'total_distance',
            'track_duration', 'mean_area', 'area_change_rate', 'mean_perimeter',
            'perimeter_change_rate', 'mean_detection_score', 'min_detection_score',
            'detection_score_std'
        ]
        
        # Convert categories to numeric labels
        unique_categories = sorted(self.features_df['category'].unique())
        self.class_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
        print(f"Classes found: {self.class_to_idx}")
        
        # Scale features
        self.scaler = StandardScaler()
        self.features_df[self.feature_columns] = self.scaler.fit_transform(
            self.features_df[self.feature_columns].astype(np.float32)
        )
    
    def __len__(self):
        return len(self.features_df)
    
    def __getitem__(self, idx):
        row = self.features_df.iloc[idx]
        
        # Get features
        features = torch.tensor(
            row[self.feature_columns].values.astype(np.float32),
            dtype=torch.float32
        )
        
        # Get label
        label = torch.tensor(
            self.class_to_idx[row['category']], 
            dtype=torch.long
        )
        
        return {
            'features': features,
            'label': label
        }

class MotionFeatureNet(nn.Module):
    def __init__(self, input_dim, num_classes, config):
        super().__init__()
        
        hidden_dims = config['model']['hidden_dims']
        dropout_rate = config['model']['dropout_rate']
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, val_loader, config, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['training']['scheduler_config']['T_0'],
        T_mult=config['training']['scheduler_config']['T_mult'],
        eta_min=config['training']['scheduler_config']['eta_min']
    )
    
    best_val_acc = 0.0
    training_history = {
        'train_losses': [],
        'train_accs': [],
        'val_accs': [],
        'learning_rates': []
    }
    
    for epoch in range(config['training']['num_epochs']):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["num_epochs"]}')
        for batch in pbar:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(features)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Update learning rate
        scheduler.step()
        
        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total
        
        # Update history
        training_history['train_losses'].append(train_loss)
        training_history['train_accs'].append(train_acc)
        training_history['val_accs'].append(val_acc)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'feature_columns': train_loader.dataset.feature_columns,
                'class_to_idx': train_loader.dataset.class_to_idx,
                'scaler': train_loader.dataset.scaler
            }, 'best_motion_model.pth')
    
    return model, training_history

def plot_training_history(history):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    epochs = range(1, len(history['train_accs']) + 1)
    
    # Accuracy plot
    ax1.plot(epochs, history['train_accs'], 'b-', label='Training Accuracy')
    ax1.plot(epochs, history['val_accs'], 'r-', label='Validation Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(epochs, history['train_losses'], 'g-')
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    # Learning rate plot
    ax3.plot(epochs, history['learning_rates'], 'm-')
    ax3.set_title('Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('motion_model_history.png')
    plt.close()

def main():
    # Set random seeds
    torch.manual_seed(CONFIG['training']['random_seed'])
    np.random.seed(CONFIG['training']['random_seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = MotionFeatureDataset(CONFIG, is_train=True)
    
    # Split data
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=CONFIG['training']['val_split'],
        random_state=CONFIG['training']['random_seed'],
        stratify=dataset.features_df['category']
    )
    
    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=CONFIG['training']['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=CONFIG['training']['batch_size'],
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Get input dimension and number of classes
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['features'].shape[1]
    num_classes = len(dataset.class_to_idx)
    
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    model = MotionFeatureNet(input_dim, num_classes, CONFIG).to(device)
    
    # Train model
    model, history = train_model(model, train_loader, val_loader, CONFIG, device)
    
    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main()