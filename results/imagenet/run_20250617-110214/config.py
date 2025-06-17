# Automatically generated run configuration (dicts and basic variables only)

model_cfg = {'type': 'EuroSATModel', 'backbone_cfg': {'type': 'ResNet', 'idims': 3, 'odims': 64, 'base_dims': 12, 'arch': [2, 2, 2, 2], 'dropout': 0.2}, 'head_cfg': {'type': 'FFN', 'idims': 64, 'odims': 200, 'hidden_dims': 1024, 'nlayers': 6, 'dropout': 0.2}}
train_dataloader = {'batch_size': 512, 'num_workers': 4, 'shuffle': True, 'drop_last': True}
val_dataloader = {'batch_size': 1, 'num_workers': 4, 'shuffle': False, 'drop_last': False}
test_dataloader = {'batch_size': 1, 'num_workers': 4, 'shuffle': False, 'drop_last': False}
optim_cfg = {'type': 'Adam', 'lr': 0.001, 'weight_decay': 0.0001}
seed = 3891198466
patience = 15
abort_condition = 0.05
direction = 'min'
metric = 'val_loss'
save_best = True
best_model_path = None
save_dir = 'results/imagenet/run_20250617-110214'
