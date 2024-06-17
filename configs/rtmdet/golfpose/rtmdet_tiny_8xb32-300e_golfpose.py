_base_ = './rtmdet_s_8xb32-300e_golfpose.py'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(in_channels=96, feat_channels=96, exp_on_reg=False))

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CachedMosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5),
    dict(type='PackDetInputs')
]

train_ezgolf = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='ezgolf/task_20240418/annotations/detection/train.json',
    data_prefix=dict(img='ezgolf/task_20240418/images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args
)

train_golfdb = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='golfdb/annotations/detection/train.json',
    data_prefix=dict(img='golfdb/images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args
)

train_dataloader = dict(
    batch_size=32,
    num_workers=10,
    #num_batch_per_epoch=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None, #dict(type='AspectRatioBatchSampler'),
    pin_memory=True,
    dataset=dict(
        type='ConcatDataset',
        datasets=[train_ezgolf, train_golfdb]
    )
)
