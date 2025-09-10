import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")

from prompts.imagenet_template import *

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS

import torch.nn.functional as F

from open_clip import tokenizer, create_model
from BLIP.models.blip_retrieval import blip_retrieval
import gem
from simfeatup_dev.upsamplers import get_upsampler

from collections import OrderedDict
import math
from typing import Callable, Optional, Sequence, Tuple
from functools import partial

from torch.utils.checkpoint import checkpoint

#from .utils import to_2tuple
#from pos_embed.py import get_2d_sincos_pos_embed

@MODELS.register_module()
class SegEarthSegmentation(BaseSegmentor):
    def __init__(self,
                 clip_type,
                 vit_type,
                 model_type,
                 name_path,
                 device=torch.device('cuda'),
                 ignore_residual=True,
                 intermediate_fusion=True,
                 attention_bias=False,
                 prob_thd=0.0,
                 logit_scale=50,
                 slide_stride=112,
                 slide_crop=224,
                 cls_token_lambda=0,
                 lambda_global=0,
                 lambda_local=0.01,
                 gaussian_std=5,
                 fusion_weight=-0.5,
                 bg_idx=0,
                 feature_up=True,
                 feature_up_cfg=dict(
                     model_name='jbu_one',
                     model_path='your/model/path'),
                 ):
        #在模型的生命周期中（例如调用 predict 或 forward 方法时），
        # MMSegmentation 会通过 data_preprocessor 自动对输入图像进行预处理（例如归一化、通道转换）
        
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True,
            size=(448, 448)
            )
        
        super().__init__(data_preprocessor=data_preprocessor)
        if clip_type == 'CLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/16', pretrained='openai', precision='fp32')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='openai', precision='fp32')
        elif clip_type == 'RemoteCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', pretrained='checkpoint/RemoteCLIP-ViT-B-32.pt', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='checkpoint/RemoteCLIP-ViT-L-14.pt', precision='fp16')
        elif clip_type == 'GeoRSCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', pretrained='checkpoint/RS5M_ViT-B-32.pt', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='checkpoint/RS5M_ViT-L-14.pt', precision='fp16')
            elif 'H' in vit_type:
                self.net = create_model('ViT-H-14', pretrained='checkpoint/RS5M_ViT-H-14.pt', precision='fp16')
        elif clip_type == 'SkyCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', \
                                        pretrained='checkpoint/SkyCLIP_ViT_B32_top50pct/epoch_20.pt', \
                                        precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', \
                                        pretrained='checkpoint/SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt', \
                                        precision='fp16')
        elif clip_type == 'OpenCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/16', pretrained='laion2b_s34b_b88k', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='laion2b_s32b_b82k', precision='fp16')
        elif clip_type == 'MetaCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B-16-quickgelu', pretrained='metaclip_fullcc', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L/14-quickgelu', pretrained='metaclip_fullcc', precision='fp16')
        elif clip_type == 'BLIP':
            if 'B' in vit_type:
                self.net = blip_retrieval(pretrained='checkpoint/model_base_14M.pth', image_size=slide_crop, vit='base')
            elif 'L' in vit_type:
                self.net = blip_retrieval(pretrained='checkpoint/model_large.pth', image_size=slide_crop, vit='large')
            #self.net = self.net.half()
        elif clip_type == 'ALIP':
            self.net = create_model('ViT-B/32', pretrained='checkpoint/ALIP_YFCC15M_B32.pt', precision='fp16')

        if model_type == 'GEM':
            if 'B' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model('ViT-B/16', 'openai', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model('ViT-B/16', 'laion2b_s34b_b88k', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'MetaCLIP':
                    self.net = gem.create_gem_model('ViT-B/16-quickgelu', 'metaclip_fullcc', ignore_residual=ignore_residual, device=device, precision='fp16')
            elif 'L' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model('ViT-L-14', 'openai', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model('ViT-L-14', 'laion2b_s32b_b82k', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'MetaCLIP':
                    self.net = gem.create_gem_model('ViT-L-14-quickgelu', 'metaclip_fullcc', ignore_residual=ignore_residual, device=device, precision='fp16')
            self.net = self.net.model

        self.net.eval().to(device)
        self.tokenizer = tokenizer.tokenize

        self.clip_type = clip_type
        self.vit_type = vit_type
        self.model_type = model_type
        self.feature_up = feature_up
        self.intermediate_fusion=intermediate_fusion
        self.attention_bias=attention_bias
        self.cls_token_lambda = cls_token_lambda
        self.lambda_global=lambda_global
        self.lambda_local=lambda_local
        self.fusion_weight=fusion_weight
        self.gaussian_std=gaussian_std
        self.output_cls_token = cls_token_lambda != 0 or lambda_global != 0,
        self.global_fusion=lambda_global != 0,
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop

        if self.clip_type == 'BLIP':
            self.patch_size = self.net.visual_encoder.patch_size
        else:
            self.patch_size = self.net.visual.patch_size

        

        #加载类别名称和索引
        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)
        #使用 openai_imagenet_template 生成文本提示，提取文本特征
        query_features = []
        with torch.no_grad(): # sub_imagenet_template, openai_imagenet_template
            for qw in query_words:
                if self.clip_type == 'BLIP':
                    query =self.net.tokenizer([temp(qw) for temp in openai_imagenet_template], padding='max_length',
                                           truncation=True, max_length=35,
                                           return_tensors="pt").to(device)
                    text_output = self.net.text_encoder(query.input_ids, attention_mask=query.attention_mask,
                                                        mode='text')
                    feature = F.normalize(self.net.text_proj(text_output.last_hidden_state[:, 0, :]))
                else:
                    query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                    feature = self.net.encode_text(query)
                    feature_time1=feature
                    feature = feature/feature_time1.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature_time2=feature
                feature = feature/feature_time2.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)

        self.dtype = self.query_features.dtype
        self.ignore_residual = ignore_residual
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        '''
        if self.intermediate_fusion:
            # 加载训练保存的 state_dict
            state_dict = torch.load("/root/autodl-tmp/zdj-SegEarth-OV/state_dict/zeroshot1/xclip_jbu_one_million_aid_attention_crf_0_tv_0.0_ent_0.0_18800.ckpt")['state_dict']

            # 只提取 fusion 模块的部分，并去掉前缀 'net.visual.fusion.'
            fusion_state_dict = {
                k.replace('model.visual.fusion.', ''): v
                for k, v in state_dict.items()
                if k.startswith('model.visual.fusion.')
            }
            # 加载到 fusion 模块中
            self.net.visual.fusion.load_state_dict(fusion_state_dict)
        '''
        '''
        if self.intermediate_fusion:
            checkpoint = torch.load("/root/autodl-tmp/zdj-SegEarth-OV/work_dirs/simfeatup_million_aid/checkpoints/jbu_one/fusion/xclip_jbu_one_million_aid_attention_crf_0_tv_0.0_ent_0.0_3400.ckpt")
            state_dict = checkpoint.get('state_dict', checkpoint)
            print("All keys in state_dict:", state_dict.keys())
            fusion_state_dict = {
                k.replace('model.visual.fusion.', ''): v
                for k, v in state_dict.items()
                if k.startswith('model.visual.fusion.')
            }
            print("Fusion keys:", fusion_state_dict.keys())
            self.net.visual.fusion.load_state_dict(fusion_state_dict)
        '''
        #初始化特征上采样器
        if feature_up:
            #借用提取的文本特征的最终维度来表示上采样之后的图像特征的维度（因为图像特征和文本特征需要对齐，所以最终的feature_dim维度是一致的）
            self.feat_dim = self.query_features.shape[-1] 
            #self.upsampler = get_upsampler(feature_up_cfg['model_name'], self.feat_dim).cuda().half()
            self.upsampler = get_upsampler(feature_up_cfg['model_name'], self.feat_dim).cuda()
            ckpt = torch.load(feature_up_cfg['model_path'])['state_dict']
            weights_dict = {k[10:]: v for k, v in ckpt.items()}
            self.upsampler.load_state_dict(weights_dict, strict=True)
        
        #只训练fusion模块
        for name, param in self.net.named_parameters():
            if 'fusion' not in name:
                param.requires_grad = False
        
        # 冻结上采样器 SimFeatUp
        if self.feature_up:
            self.upsampler.eval()
            for param in self.upsampler.parameters():
                param.requires_grad = False
    
    def forward_feature(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]
        batch_size=img.shape[0]
        img = img.cuda()
        #print(img)
        #图像特征提取
        if self.clip_type == 'BLIP':
            img = F.interpolate(img, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            image_features = self.net.visual_encoder(img, self.ignore_residual)
            image_features = self.net.vision_proj(image_features[:, 1:, ])
        elif self.model_type == 'GEM':
            image_features = self.net.visual(img)
        else:
            image_features = self.net.encode_image(img, self.model_type, self.ignore_residual, self.output_cls_token,self.intermediate_fusion,self.attention_bias,self.lambda_local,self.gaussian_std,self.fusion_weight)
        #全局偏见缓解（CLS Token 处理）
        if self.output_cls_token:
            image_cls_token, image_features = image_features
            image_cls_token_time=image_cls_token
            image_cls_token = image_cls_token/image_cls_token_time.norm(dim=-1, keepdim=True)

            if self.global_fusion:
                #特征级融合
                cls_features = image_cls_token.view(batch_size, 1, -1)  # 形状 (1, 1, feat_dim)
                image_features = image_features + self.lambda_global*cls_features  # 形状 (1, num_patches, feat_dim)
                
            #logits融合
            cls_logits = image_cls_token @ self.query_features.T

        # featup上采样
        if self.feature_up:
            #计算图像经过特征提取之后的特征图的分辨率
            feature_w, feature_h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
            image_w, image_h = img[0].shape[-2], img[0].shape[-1]
            image_features = image_features.permute(0, 2, 1).view(batch_size, self.feat_dim, feature_w, feature_h)
            #with torch.cuda.amp.autocast():
            #    image_features = self.upsampler(image_features, img).half()
            image_features = self.upsampler(image_features, img)
            image_features = image_features.view(batch_size, self.feat_dim, image_w * image_h).permute(0, 2, 1)
        image_features_time=image_features
        #image_features = self.apply_gaussian_weight_flat(feature_map_flat=image_features, gaussian_std=1.0, lambda_local=0.05)
        image_features = image_features/image_features_time.norm(dim=-1, keepdim=True)

        
        #相似性计算
        logits = image_features @ self.query_features.T

        if self.output_cls_token:
            cls_logits = cls_logits.unsqueeze(1)
            logits = logits + cls_logits * self.cls_token_lambda

            # # CLIP Surgery
            # # weights to restrain influence of obvious classes on others
            # prob = (cls_logits * 2).softmax(-1)
            # w = prob / prob.mean(-1, keepdim=True)
            # # element-wise multiplied features
            # b, n_t, n_i, c = image_features.shape[0], self.query_features.shape[0], image_features.shape[1], image_features.shape[2]
            # feats = image_features.reshape(b, n_i, 1, c) * self.query_features.reshape(1, 1, n_t, c)
            # feats *= w.reshape(1, 1, n_t, 1)
            # redundant_feats = feats.mean(2, keepdim=True) # along cls dim
            # feats = feats - redundant_feats
            # # sum the element-wise multiplied features as cosine similarity
            # logits = feats.sum(-1)
        

        if self.feature_up:
            w, h = img[0].shape[-2], img[0].shape[-1]
        else:
            w, h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(batch_size, out_dim, w, h)
        
        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')
        
        return logits
    '''
    滑动窗口推理(Sliding-Window Inference):
    用于处理大尺寸图像（例如高分辨率的遥感图像），避免直接处理整张图像带来的内存或计算限制。
    将图像分割成多个小块(crop)，逐个小块进行推理(调用 forward_feature 方法)，然后将结果拼接回原始图像尺寸。
    '''
    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        #计算窗口数量(确保覆盖整个图像)
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        #初始化预测张量和计数矩阵
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, self.patch_size[0])
                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)
                #推理当前小块
                crop_seg_logit = self.forward_feature(crop_img)

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]
                #累加预测结果
                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        #加权平均和上采样
        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        return logits
    '''
    predict 方法是推理的入口，用于对输入图像进行语义分割预测。它支持两种推理模式（滑动窗口或直接推理），
    并通过 postprocess_result 方法将 logits 转换为最终的分割掩码。
    '''
    @torch.no_grad()
    def predict(self, inputs, data_samples):
        #构造图像元信息
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]
        #inputs = inputs.half()
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

        return self.postprocess_result(seg_logits, data_samples)
    '''
    postprocess_result 方法用于将模型推理得到的 logits 转换为最终的语义分割结果（分割掩码），
    并根据 data_samples 的情况返回不同格式的结果。
    '''
    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        '''
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h
            #映射查询到类别
            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
            #生成分割掩码
            seg_pred = seg_logits.argmax(0, keepdim=True)#对类别维度（第 0 维）取最大值，生成分割掩码
            #如果最大概率低于 self.prob_thd，将该像素的类别设为背景（self.bg_idx）
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples
    '''
        for i in range(batch_size):
            logits_i = seg_logits[i] * self.logit_scale
            logits_i = logits_i.softmax(0)  # n_queries * w * h
            #映射查询到类别
            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                logits_i = logits_i.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                logits_i = (logits_i * cls_index).max(1)[0]
            #生成分割掩码
            seg_pred = logits_i.argmax(0, keepdim=True)#对类别维度（第 0 维）取最大值，生成分割掩码
            #如果最大概率低于 self.prob_thd，将该像素的类别设为背景（self.bg_idx）
            seg_pred[logits_i.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': logits_i}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples
    '''
    compute_padsize 方法计算 padding 大小，
    确保图像尺寸是 patch_size 的整数倍(ViT 模型的要求)
    '''
    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        #inputs = inputs.half()
       
        # 前向传播
        seg_logits = self.forward_feature(inputs)

        # 获取 ground truth
        seg_labels = torch.stack([ds.gt_sem_seg.data for ds in data_samples]).to(device=inputs.device, dtype=torch.long).squeeze(1)

        # 固定类权重（根据类别不平衡程度设置）
        #class_weights = torch.tensor([1.0, 1.0, 1.0, 20.0, 4.0], device=inputs.device)
        # 计算损失
        loss_ce = F.cross_entropy(seg_logits, seg_labels,ignore_index=255)
        
        losses = {'loss_ce': loss_ce}
        # 汇总损失
        losses['loss'] = sum(losses.values())
        return losses
    @staticmethod
    def gaussian_window(dim1, dim2, std=1., device='cpu'):
        constant = 1 / (std * math.sqrt(2))
        ks = list()
        for dim in [dim1, dim2]:
            start = -(dim - 1) / 2.0
            k = torch.linspace(start=start * constant,
                               end=(start + (dim - 1)) * constant,
                               steps=dim,
                               dtype=torch.float,
                               device=device)
            ks.append(k)
        dist_square_to_mu = (torch.stack(torch.meshgrid(*ks, indexing='ij')) ** 2).sum(0)
        return torch.exp(-dist_square_to_mu)

    @staticmethod
    def get_attention_addition(dim1, dim2, window):
        print("ready!")
        m = torch.einsum('ij,kl->ijkl', torch.eye(dim1, device=window.device), torch.eye(dim2, device=window.device))
        
        m = m.permute((0, 3, 1, 2)).contiguous()  # m[ijkl] = 1 iff (i, j) == (k, l)
        
        out = F.conv2d(m.view(-1, dim1, dim2).unsqueeze(1), window.unsqueeze(0).unsqueeze(1), padding='same').squeeze(1)
        print("done")
        out = out.view(dim1 * dim2, dim1 * dim2)
        return out
    def apply_gaussian_weight_flat(self, feature_map_flat, gaussian_std=5, lambda_local=0.007):
        """
        将高斯加权局部偏置应用到已展平的特征图上。
        """

        batch_size, num_tokens, d_model = feature_map_flat.size()
        side_length = int(math.sqrt(num_tokens))
        height, width = side_length, side_length
        
        # 将展平的特征图重新整形为2D图像形状
        feature_map_2d = feature_map_flat.permute(0, 2, 1).view(batch_size, d_model, height, width)

        # 1. 创建一个固定大小的高斯窗口
        # 选择一个较小的、可控的窗口大小，例如 31
        window_size = 65 
        #window_center = (window_size - 1) / 2
        # 注意：这里我们只创建一个中心化的窗口，因为它会作为卷积核使用
        window = self.gaussian_window(window_size, window_size, std=gaussian_std, device=feature_map_flat.device)
        # 将窗口 reshape 成适合conv2d的卷积核形状 (out_channels, in_channels, kernel_h, kernel_w)
        # 由于我们对每个特征通道应用相同的权重，所以输入和输出通道都是d_model
        # 注意：这里可能需要根据你的具体模型和需求调整
        window = window.view(1, 1, window_size, window_size).repeat(d_model, 1, 1, 1)

        # 2. 应用高斯卷积
        # 使用分组卷积（groups=d_model）对每个通道独立进行卷积，以实现局部加权
        # 这里的卷积操作将每个像素的特征值与其局部邻域内的加权值进行求和
        weighted_features_2d = F.conv2d(feature_map_2d, window, padding='same', groups=d_model)

        # 3. 将加权后的特征与原始特征相加
        output_feature_map_2d = feature_map_2d + weighted_features_2d * lambda_local

        # 4. 将输出特征图重新展平回 (B, H*W, C) 形状
        output_feature_map_flat = output_feature_map_2d.view(batch_size, d_model, -1).permute(0, 2, 1)

        return output_feature_map_flat
    
    def apply_gaussian_weight_flat_origin(self,feature_map_flat, gaussian_std=5, lambda_local=0.007):
        """
        将高斯加权局部偏置应用到已展平的特征图上。

        参数:
            feature_map_flat (torch.Tensor): 已展平的输入特征图张量，形状为 (B, H*W, C)。
                                            例如：torch.Size([1, 50176, 512])
            gaussian_std (float): 高斯窗口的标准差，控制局部加权的范围。
            lambda_local (float): 局部偏置的权重因子，控制其对最终特征图的影响程度。

        返回:
            torch.Tensor: 经过高斯加权后，带有局部偏置的特征图。
        """
        # 确保输入是3D张量
        if feature_map_flat.dim() != 3:
            raise ValueError("输入特征图必须是形状为 (B, H*W, C) 的3D张量")

        batch_size, num_tokens, d_model = feature_map_flat.size()
        
        # 从 num_tokens (H*W) 推断出原始的 height 和 width
        side_length = int(math.sqrt(num_tokens))
        
        height, width = side_length, side_length
        
        # 1. 生成高斯偏置矩阵 (omega)
        # 调用你提供的函数来生成高斯窗口和偏置矩阵
        window = SegEarthSegmentation.gaussian_window(height * 2 - 1, width * 2 - 1, std=gaussian_std,device=feature_map_flat.device)
        print(f"window device: {window.device}") # 添加这行来检查
        # omega 的计算是耗时步骤，在这里打印以确认数据设备
        print(f"开始计算 omega，输入张量设备：{feature_map_flat.device}")
        omega = SegEarthSegmentation.get_attention_addition(height, width, window)
        
        print(omega.shape)
        print(f"结束计算 omega：{omega.device}")
        # 2. 将偏置矩阵的形状扩展以匹配批次大小，并移动到正确的设备
        # omega 的形状为 (1, H*W, H*W)
        omega = omega.unsqueeze(0).to(feature_map_flat.dtype).to(feature_map_flat.device)

        # 3. 应用高斯偏置
        # 我们使用批量矩阵乘法来应用偏置。这相当于对每个 token 的特征向量
        # 进行加权求和，权重由 omega 矩阵提供。
        # torch.bmm(omega, feature_map_flat) 的形状是 (B, H*W, H*W) @ (B, H*W, C) -> (B, H*W, C)
        weighted_x = torch.bmm(omega.expand(batch_size, -1, -1), feature_map_flat)
        
        # 4. 将加权后的偏置特征加到原始特征图上
        # 通过 lambda_local 来控制偏置的影响程度
        output_feature_map = feature_map_flat + weighted_x * lambda_local

        return output_feature_map
    

'''
get_cls_idx 是一个独立函数，用于从文件中读取类别名称和对应的类别索引，
生成类别名称列表和索引列表
'''
def get_cls_idx(path):
    #打开文件，读取所有行。name_sets：每行是一个字符串，表示一个类别的名称集合
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)
    #解析类别名称和索引
    # 结果：class_names = ['forest', 'woods\n', 'road', 'highway\n', 'river\n']
    #class_indices = [0, 0, 1, 1, 2]
    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices