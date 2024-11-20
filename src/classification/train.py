import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import torchvision.models as models

# Configuration dictionary
CONFIG = {
    'datasets': ['M1', 'M2'],
    'data_config': {
        # 'MDA': {
        #     'feature_file': '/scratch/rl4789/CellClassif/data/processed/classification/MDA/all_features.csv',
        #     'class_name': 'MDA'
        # },
        # 'FB': {
        #     'feature_file': '/scratch/rl4789/CellClassif/data/processed/classification/FB/all_features.csv',
        #     'class_name': 'FB'
        # },
        'M1': {
            'feature_file': '/scratch/rl4789/CellClassif/data/processed/classification/M1/all_features.csv',
            'class_name': 'M1'
        },
        'M2': {
            'feature_file': '/scratch/rl4789/CellClassif/data/processed/classification/M2/all_features.csv',
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
            'type': 'cosine_warmup_restarts',
            'T_0': 50,
            'T_mult': 1,
            'eta_min': 1e-6,
            'warmup_epochs': 5
        },
        'feature_selection': {
            'enabled': True,
            'num_features': 20
        }
    }
}

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

class CellDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.config = config
        self.is_train = is_train
        print("Initializing CellDataset...")
        
        # Load all features
        features_list = []
        for dataset_name in config['datasets']:
            try:
                df = pd.read_csv(config['data_config'][dataset_name]['feature_file'])
                
                # Handle first column as track_id if unnamed
                if 'Unnamed: 0' in df.columns:
                    df = df.rename(columns={'Unnamed: 0': 'track_id'})
                
                # Ensure proper data types
                df['track_id'] = df['track_id'].astype(int)
                df['category'] = config['data_config'][dataset_name]['class_name']
                df['video_id'] = df['video_id'].astype(str)
                
                # Ensure all feature columns are float32
                feature_cols = [col for col in df.columns 
                              if col not in ['track_id', 'category', 'video_id']]
                for col in feature_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
                
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
        
        print(f"Total samples: {len(self.features_df)}")
        
        # Initialize feature columns (excluding non-feature columns)
        self.feature_columns = [col for col in self.features_df.columns 
                              if col not in ['track_id', 'category', 'video_id']]
        
        # Convert categories to numeric labels
        unique_categories = sorted(self.features_df['category'].unique())
        self.class_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
        print(f"Classes found: {self.class_to_idx}")
        
        # Feature selection if enabled and in training mode
        if is_train and config['training'].get('feature_selection', {}).get('enabled', False):
            self.select_important_features(
                k=config['training']['feature_selection'].get('num_features', 20)
            )
            print(f"Selected {len(self.feature_columns)} features")
        
        # Scale features
        self.scaler = StandardScaler()
        self.features_df[self.feature_columns] = self.scaler.fit_transform(
            self.features_df[self.feature_columns].astype(np.float32)
        )
        
        # Verify data types
        print("\nFeature data types:")
        print(self.features_df[self.feature_columns].dtypes)
    
    
    def select_important_features(self, k=20):
        """
        Select the k most important features using Random Forest feature importance
        """
        from sklearn.ensemble import RandomForestClassifier
        
        print("\nPerforming feature selection...")
        X = self.features_df[self.feature_columns].values
        y = self.features_df['category'].map(self.class_to_idx).values
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance scores
        importance = rf.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        # Select top k features
        self.selected_features = [self.feature_columns[i] for i in indices[:k]]
        self.feature_columns = self.selected_features
        
        print("\nTop features and their importance scores:")
        for f, imp in zip(self.selected_features, importance[indices[:k]]):
            print(f'{f}: {imp:.4f}')
    
    def get_sampler(self):
        """
        Get weighted sampler for balanced training
        """
        class_counts = self.features_df['category'].value_counts()
        weights = [1.0/class_counts[self.features_df.iloc[idx]['category']] 
                  for idx in range(len(self.features_df))]
        return WeightedRandomSampler(weights, len(weights))
    
    def __len__(self):
        return len(self.features_df)
    
    def __getitem__(self, idx):
        row = self.features_df.iloc[idx]
        
        features = torch.tensor(
            row[self.feature_columns].values.astype(np.float32),
            dtype=torch.float32
        )
        
        label = torch.tensor(
            self.class_to_idx[row['category']], 
            dtype=torch.long
        )
        
        return {
            'features': features,
            'label': label
        }

class SimpleCellNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class ResNetCellClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, backbone='resnet18'):
        super().__init__()
        
        # Create feature vector to 2D image transformation
        self.reshape_layer = nn.Sequential(
            nn.Linear(input_dim, 32 * 32),
            nn.ReLU(),
            nn.BatchNorm1d(32 * 32)
        )
        
        # Initialize ResNet backbone
        if backbone == 'resnet18':
            self.resnet = models.resnet18(weights=None)
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(weights=None)
        
        # Modify first conv layer to accept single-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Reshape feature vector to 2D image
        x = self.reshape_layer(x)
        x = x.view(-1, 1, 32, 32)
        return self.resnet(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    """
    Train the model with MixUp augmentation
    """
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_accs = []
    learning_rates = []
    
    def mixup_data(x, y, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            mixed_features, labels_a, labels_b, lam = mixup_data(features, labels)
            
            outputs = model(mixed_features)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            with torch.no_grad():
                orig_outputs = model(features)
                _, predicted = orig_outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{current_lr:.6f}'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_accs.append(val_acc)
        
        scheduler.step()
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'feature_columns': train_loader.dataset.feature_columns,  # Save selected features
                'class_to_idx': train_loader.dataset.class_to_idx,  # Save class mapping
            }, 'best_model.pth')
    
    return train_losses, train_accs, val_accs, learning_rates

def main():
    # Set random seeds
    torch.manual_seed(CONFIG['training']['random_seed'])
    np.random.seed(CONFIG['training']['random_seed'])
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load and split data
    full_dataset = CellDataset(CONFIG, is_train=True)
    
    # Split indices
    indices = np.arange(len(full_dataset))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=CONFIG['training']['val_split'],
        random_state=CONFIG['training']['random_seed'],
        stratify=full_dataset.features_df['category']
    )
    
    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        full_dataset,
        batch_size=CONFIG['training']['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        full_dataset,
        batch_size=CONFIG['training']['batch_size'],
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Get input dimension and number of classes
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['features'].shape[1]
    num_classes = len(full_dataset.class_to_idx)
    
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    # model = SimpleCellNet(input_dim, num_classes).to(device
    model = ResNetCellClassifier(input_dim, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['training']['learning_rate'],
        weight_decay=CONFIG['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CONFIG['training']['scheduler_config']['T_0'],
        T_mult=CONFIG['training']['scheduler_config']['T_mult'],
        eta_min=CONFIG['training']['scheduler_config']['eta_min']
    )
    
    # Train model
    train_losses, train_accs, val_accs, learning_rates = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=CONFIG['training']['num_epochs'],
        device=device
    )
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {max(val_accs):.2f}%")
    
    # Plot training history
    plot_training_history(train_losses, train_accs, val_accs, learning_rates)

def plot_training_history(train_losses, train_accs, val_accs, learning_rates):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    epochs = range(1, len(train_accs) + 1)
    
    # Accuracy plot
    ax1.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax1.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(epochs, train_losses, 'g-')
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    # Learning rate plot
    ax3.plot(epochs, learning_rates, 'm-')
    ax3.set_title('Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    main()