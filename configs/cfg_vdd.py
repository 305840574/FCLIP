_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_vdd.txt',
    prob_thd=0.3,
    lambda_global=-1.1,
    lambda_local=-0.18,
    gaussian_std=0.9,
    fusion_weight=-0.1,
    #model_type='SegEarth',
    #ignore_residual=False,
    #cls_token_lambda=0,
    #feature_up=False,
    #intermediate_fusion=False
)

# dataset settings
dataset_type = 'VDDDataset'
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
            img_path='data/VDD/train/src',
            seg_map_path='data/VDD/train/gt'),
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
            img_path='data/VDD/val/src',
            seg_map_path='data/VDD/val/gt'),
        pipeline=val_pipeline
    )
)
#zdj

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='data/VDD/test/src',
            seg_map_path='data/VDD/test/gt'),
        pipeline=test_pipeline))

# 训练配置(zdj)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=20)  # 50个周期，每5个周期验证一次
val_cfg = dict(type='ValLoop')
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.001),   
)
# 学习率调度器
param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[120], gamma=0.2)
#zdj