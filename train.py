import src

from src.runner import Runner
device = 'cuda:0'


# ----------------------------------
# Base Configurations
# ----------------------------------

loading_cfg = dict(
    batch_size=512,
    num_workers=4,
)

eurosat_cfg = dict(
    type='EuroSATDataset',
    transform=[
        dict(type='Resize', size=(128, 128)),
        dict(type='ToTensor'),
    ]
)

imagenet_cfg = dict(
    type='ImageNetDataset',
    transform=[
        dict(type='Resize', size=(128, 128)),
        dict(type='ToTensor'),
    ]
)

optim_cfg = dict(
    type='Adam',
    lr=0.001,
    weight_decay=1e-4,
)

backbone_cfg = dict(
    type='ResNet',
    idims=3,
    odims=64,
    base_dims=12,
    arch=[2, 2, 2, 2],
    dropout=0.2,
)

# ------------------------
# EuroSAT Model Configuration
# ------------------------


# -----------------------------------
# Vanilla EuroSAT model configuration
# -----------------------------------

eurosat_model_cfg = dict(
    type='EuroSATModel',
    backbone_cfg=backbone_cfg,
    head_cfg=dict(
        type='FFN',
        idims=64,
        odims=10,  # EuroSAT has 10 classes
        hidden_dims=1024,
        nlayers=6,
        dropout=0.2,
    )
)
# -----------------------------------
# Tiny ImageNet model configuration
# -----------------------------------
imagenet_model_cfg = dict(
    type='EuroSATModel',
    backbone_cfg=backbone_cfg,
    head_cfg=dict(
        type='FFN',
        idims=64,
        odims=200,  # Tiny ImageNet has 200 classes
        hidden_dims=1024,
        nlayers=6,
        dropout=0.2,
    )
)

# -----------------------------------
# TFL EuroSAT model configuration
# -----------------------------------
tfl_eurosat_model_cfg = dict(
    type='EuroSATModel',
    backbone_cfg=backbone_cfg,
    head_cfg=dict(
        type='FFN',
        idims=64,
        odims=10,  # EuroSAT has 10 classes
        hidden_dims=1024,
        nlayers=6,
        dropout=0.2,
    ),
    ckpt=dict(
        path = '/mmdetection3d/PRIVATE/MLSP/results/tiny_imnet/run_20250613-065115/best_model.pth',
        load_head=False,
        load_backbone=True,
        strict=True,
    )
)

runner = Runner(model=tfl_eurosat_model_cfg, dataloader_cfg=loading_cfg, dataset=eurosat_cfg, optim=optim_cfg, device=device, work_dir='results/eurosat_tfl')
runner.run(mode='train', val_interval=1, log_interval=1, epochs=100, start_epoch=1)
