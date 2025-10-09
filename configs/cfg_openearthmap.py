_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_openearthmap.txt',
    prob_thd=0.1,
    lambda_global=0.007,
    lambda_local=-1.6,
    gaussian_std=5,
    fusion_weight=-0.5,
)

# dataset settings
dataset_type = 'OpenEarthMapDataset'
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

