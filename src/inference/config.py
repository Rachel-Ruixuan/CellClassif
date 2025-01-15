# config.py

TRAINING_CONFIG = {
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
