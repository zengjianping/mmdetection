_base_ = [
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_1x.py',
    './rtmdet_tta.py'
]

model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1,
        widen_factor=1,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
)

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
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

val_ezgolf = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='ezgolf/task_20240418/annotations/detection/val.json',
    data_prefix=dict(img='ezgolf/task_20240418/images/'),
    test_mode=True,
    pipeline=test_pipeline,
    backend_args=backend_args
)

val_golfdb = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='golfdb/annotations/detection/val.json',
    data_prefix=dict(img='golfdb/images/'),
    test_mode=True,
    pipeline=test_pipeline,
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

val_dataloader = dict(
    batch_size=4,
    num_workers=10,
    #num_batch_per_epoch=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[val_ezgolf, val_golfdb]
    )
)
test_dataloader = val_dataloader

val_evaluator_ezgolf = dict(
    type='CocoMetric',
    ann_file=data_root + 'ezgolf/task_20240418/annotations/detection/val.json',
    metric='bbox',
    format_only=False,
    proposal_nums=(100, 1, 10),
    backend_args=backend_args
)

val_evaluator_golfdb = dict(
    type='CocoMetric',
    ann_file=data_root + 'golfdb/annotations/detection/val.json',
    metric='bbox',
    format_only=False,
    proposal_nums=(100, 1, 10),
    backend_args=backend_args
)

val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[val_evaluator_ezgolf, val_evaluator_golfdb],
    dataset_prefixes=['ezgolf', 'golfdb'])
test_evaluator = val_evaluator

# training
max_epochs = 100
stage2_num_epochs = 80
switch_epochs = max_epochs - stage2_num_epochs
base_lr = 0.004
interval = 5

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=switch_epochs,
        switch_pipeline=train_pipeline_stage2)
]
