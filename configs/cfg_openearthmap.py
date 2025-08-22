_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_openearthmap.txt',
    prob_thd=0.1,
    #feature_cls_token_lambda=-1.6,
    cls_token_lambda=-0.3,
    lambda_local=0.01,
    gaussian_std=5,
    model_type='SegEarth'
)

# dataset settings
dataset_type = 'OpenEarthMapDataset'
data_root = ''

#zdj
# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# Validation/Test pipeline
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# Training dataloader
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='data/OpenEarthMap/img_dir/train',
            seg_map_path='data/OpenEarthMap/ann_dir/train'),
        pipeline=train_pipeline
    )
)

# Validation dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='data/OpenEarthMap/img_dir/val',
            seg_map_path='data/OpenEarthMap/ann_dir/val'),
        pipeline=val_pipeline
    )
)
#zdj

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(990, 990), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='data/OpenEarthMap/img_dir/val',
            seg_map_path='data/OpenEarthMap/ann_dir/val'),
        pipeline=test_pipeline))

# 训练配置(zdj)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=150, val_interval=1)  # 50个周期，每5个周期验证一次
val_cfg = dict(type='ValLoop')
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.001),   
)
# 学习率调度器
param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[150], gamma=0.2)
#zdj