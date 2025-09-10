_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_iSAID.txt',
    prob_thd=0.4,
    #lambda_global=-1.6,
    cls_token_lambda=-0.3,
    lambda_local=0.01,
    gaussian_std=3,
    model_type='SegEarth',
)

# dataset settings
dataset_type = 'iSAIDDataset'
data_root = ''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(660, 660), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='data/iSAID/img_dir/val',
            seg_map_path='data/iSAID/ann_dir/val'),
        pipeline=test_pipeline))
