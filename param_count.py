from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from segmentor import Segmentation
from collections import OrderedDict
import numpy as np
def sizeof_fmt(num_bytes):
    
    for unit in ['B','KB','MB','GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"

def summarize_model_params_total(model, dtype_size=4):

    total_params = sum(p.numel() for p in model.parameters())
    total_size = total_params * dtype_size / 1024**2 
    print(f"params: {total_params:,}")
    print(f"model size: {total_size:.2f} MB")
    return total_params, total_size


model = Segmentation(
    clip_type='CLIP',     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
    vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
    model_type='MaskCLIP',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth','FCLIP'
    ignore_residual=True,  # introduced from ClearCLIP; optional in SegEarth/FCLIP
    intermediate_fusion=True,  # controls fusion of the I (Low-Level) branch
    global_fusion=True, # controls fusion of the G (Global) branch
    attention_bias=False,  # for other methods to toggle L (Local) branch fusion;
                           # fused by default in FCLIP
    feature_up=True,  # introduced in SegEarth; optional in FCLIP
    feature_up_cfg=dict(
        model_name='jbu_one',
        model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
    cls_token_lambda= -0.3, # introduced in SegEarth; optional in FCLIP
    lambda_global= 0,
    lambda_local=0.01,
    gaussian_std=5,
    name_path='./configs/my_name.txt',
    fusion_weight=0,
)
summarize_model_params_total(model)
