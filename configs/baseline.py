dataset = 'EuroSATDataset'
batch_size = 512

optim_cfg = dict(
    type='Adam',
    lr=0.001,
    weight_decay=1e-4,
)

backbone_cfg = dict(
    type='ResNet',
    idims=3,
    odims=10,
    base_dims=12,
    arch=[2, 2, 2, 2],
    dropout=0.2,
)

train_dataloader = dict(
    pipeline=[
        dict(type='Resize', size=(128, 128)),
        dict(type='RandAugment', num_ops=2, magnitude=9),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    batch_size=batch_size,
    num_workers=4,
    shuffle=True,
    drop_last=True,
)

val_dataloader = dict(
    pipeline =[
        dict(type='Resize', size=(128, 128)),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    batch_size=1,
    num_workers=4,
    shuffle=False,
    drop_last=False,
)

test_dataloader = val_dataloader.copy()

model_cfg = dict(
    type='Baseline',
    backbone_cfg=backbone_cfg,
)

runner_args = dict(
    metric = 'mean_ap',
    direction = 'max',
    patience = 15,
)
work_dir = 'results/eurosat/'
device = 'cuda:0'
