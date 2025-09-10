# base configurations
model = dict(
    type='SegEarthSegmentation',
    clip_type='CLIP',     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
    vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
    model_type='FCLIP',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth','FCLIP'
    ignore_residual=True,
    intermediate_fusion=True,
    attention_bias=False,#FCLIP uses attention_bias by default,this parameter is provided for use by other models.
    feature_up=True,
    feature_up_cfg=dict(
        model_name='jbu_one',
        model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
        #model_path='/root/autodl-tmp/zdj-SegEarth-OV/work_dirs/simfeatup_million_aid/checkpoints/jbu_one/xclip_jbu_one_million_aid_attention_crf_0_tv_0.0_ent_0.0_5200.ckpt'),
    cls_token_lambda= -0.3,
    lambda_global= 0,
    lambda_local=0.01,
    gaussian_std=5,
    fusion_weight=0,
)

# 评估器

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, alpha=0.5, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')
log_level = 'DEBUG'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook',
        interval=1000,
        save_best='mIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1))

custom_hooks = [
    dict(
        type='SaveTrainableModulesHook',
        interval=1,  # 每 5 轮触发
        metric_key='mIoU',  # 验证指标名称
        rule='greater',     # 越大越好
    )
]
