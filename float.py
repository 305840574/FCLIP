from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from segearth_segmentor import SegEarthSegmentation
from collections import OrderedDict
import numpy as np
def sizeof_fmt(num_bytes):
    """将字节数格式化为 MB/GB"""
    for unit in ['B','KB','MB','GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"

def summarize_model_params_total(model, dtype_size=4):
    """
    统计模型总参数量和存储大小
    dtype_size: 每个参数占用字节，默认 float32=4
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_size = total_params * dtype_size / 1024**2  # 转换为 MB
    print(f"模型总参数量: {total_params:,}")
    print(f"模型总存储大小: {total_size:.2f} MB")
    return total_params, total_size


model = SegEarthSegmentation(
    #type='SegEarthSegmentation',
    clip_type='CLIP',     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
    vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
    model_type='MaskCLIP',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth','FCLIP'
    #ignore_residual=True,
    #intermediate_fusion=True,
    attention_bias=False,#FCLIP uses attention_bias by default,this parameter is provided for use by other models.
    feature_up=False,
    feature_up_cfg=dict(
        model_name='jbu_one',
        model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
        #model_path='/root/autodl-tmp/zdj-SegEarth-OV/work_dirs/simfeatup_million_aid/checkpoints/jbu_one/xclip_jbu_one_million_aid_attention_crf_0_tv_0.0_ent_0.0_5200.ckpt'),
    cls_token_lambda= 0.0,
    lambda_global= 0,
    lambda_local=0.01,
    gaussian_std=5,
    name_path='./configs/my_name.txt',
    fusion_weight=0,
)
summarize_model_params_total(model)
