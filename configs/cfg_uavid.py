_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_uavid.txt',
    prob_thd=0.3,
    feature_cls_token_lambda=2,
    cls_token_lambda=-0.45,
    lambda_local=0.02,
    gaussian_std=5,
    #model_type='SegEarth'
)

# dataset settings
dataset_type = 'UAVidDataset'
data_root = ''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(850, 850), keep_ratio=True),
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
            img_path='data/UAVid/img_dir/test',
            seg_map_path='data/UAVid/ann_dir/test'),
        pipeline=test_pipeline))