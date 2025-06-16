
loading_cfg = dict(
    batch_size=512,
    num_workers=4,
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

dataset_cfg = dict(
    type='ImageNetDataset',
    transform=[
        dict(type='Resize', size=(128, 128)),
        dict(type='AutoAugment', policies='imagenet'),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model_cfg = dict(
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