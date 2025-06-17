# Automatically generated run configuration (dicts and basic variables only)

model_cfg = {'type': 'EuroSATModel', 'backbone_cfg': {'type': 'ResNet', 'idims': 3, 'odims': 512, 'base_dims': 32, 'arch': [2, 2, 2, 2], 'dropout': 0.2}, 'head_cfg': {'type': 'FFN', 'idims': 512, 'odims': 10, 'hidden_dims': 1024, 'nlayers': 6, 'dropout': 0.2}, 'ckpt': {'path': 'results/imagenet_large/run_20250617-152539/best_model.pth', 'load_head': False, 'load_backbone': True, 'strict': True}}
train_dataloader = {'batch_size': 512, 'num_workers': 4, 'shuffle': True, 'drop_last': True}
val_dataloader = {'batch_size': 1, 'num_workers': 4, 'shuffle': False, 'drop_last': False}
test_dataloader = {'batch_size': 1, 'num_workers': 4, 'shuffle': False, 'drop_last': False}
optim_cfg = {'type': 'Adam', 'lr': 0.001, 'weight_decay': 0.0001}
seed = 567508249
patience = 15
abort_condition = 0.05
direction = 'max'
metric = 'mean_ap'
save_best = True
best_model_path = None
save_dir = 'results/eurosat_large_tl/run_20250617-201723'
