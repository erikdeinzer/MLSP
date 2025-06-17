# Automatically generated run configuration (dicts and basic variables only)

model_cfg = {'type': 'Baseline', 'backbone_cfg': {'type': 'ResNet', 'idims': 3, 'odims': 10, 'base_dims': 12, 'arch': [2, 2, 2, 2], 'dropout': 0.2}}
train_dataloader = {'batch_size': 512, 'num_workers': 4, 'shuffle': True, 'drop_last': True}
val_dataloader = {'batch_size': 1, 'num_workers': 4, 'shuffle': False, 'drop_last': False}
test_dataloader = {'batch_size': 1, 'num_workers': 4, 'shuffle': False, 'drop_last': False}
optim_cfg = {'type': 'Adam', 'lr': 0.001, 'weight_decay': 0.0001}
seed = 44809880
patience = 15
abort_condition = 0.05
direction = 'max'
metric = 'mean_ap'
save_best = True
best_model_path = None
save_dir = 'results/eurosat/run_20250617-104201'
