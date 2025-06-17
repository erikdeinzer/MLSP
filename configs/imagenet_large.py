from torchvision.transforms import AutoAugmentPolicy

batch_size = 512
dataset = 'ImageNetDataset'
device = 'cuda:0'
optim_cfg = dict(
    type='Adam',
    lr=0.001,
    weight_decay=1e-4,
)

backbone_cfg = dict(
    type='ResNet',
    idims=3,
    odims=512,
    base_dims=32,
    arch=[2, 2, 2, 2],
    dropout=0.2,
)

train_dataloader = dict(
    pipeline=[
        dict(type='Resize', size=(128, 128)),
        dict(type='AutoAugment', policy=AutoAugmentPolicy.IMAGENET),
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




work_dir = 'results/imagenet_large_tl/'

model_cfg = dict(
    type='EuroSATModel',
    backbone_cfg=backbone_cfg,
    head_cfg=dict(
        type='FFN',
        idims=512,
        odims=200,  # Tiny ImageNet has 200 classes
        hidden_dims=1024,
        nlayers=6,
        dropout=0.2,
    ),
    ckpt=dict(
        path = 'results/imagenet_large/run_20250617-152539/best_model.pth', # Can be replaced with a path to a pre-trained model
        load_head=False,
        load_backbone=True,
        strict=True,
    )
)