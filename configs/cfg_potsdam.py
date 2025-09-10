_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_potsdam.txt',
    prob_thd=0.1,
    bg_idx=5,
    lambda_global=5,
    cls_token_lambda= -0.5,
    lambda_local=0.015,
    gaussian_std=5,
    fusion_weight=-0.1,
    #ignore_residual=False,
    #model_type='SegEarth',
    #intermediate_fusion=True
)

# dataset settings
dataset_type = 'PotsdamDataset'
data_root = ''

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
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='data/potsdam/img_dir/val',
            seg_map_path='data/potsdam/ann_dir/val'),
        pipeline=test_pipeline))