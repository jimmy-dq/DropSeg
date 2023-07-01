# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from tkinter.messagebox import NO

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.layers.helpers import *
from timm.models.vision_transformer import PatchEmbed
from model.network import Decoder
import torch.nn.functional as F


def interpolate_pos_embed(pos_embed, search_size):
    
        num_extra_tokens = 1
        # pos_embed = net.pos_embed
        model_pos_tokens = pos_embed[:, num_extra_tokens:, :] # bs, N, C
        model_token_size = int(model_pos_tokens.shape[1]**0.5)
        extra_pos_tokens = pos_embed[:, :num_extra_tokens]

        embedding_size = extra_pos_tokens.shape[-1]

        if search_size != model_token_size: # do interpolation
            model_pos_tokens_temp = model_pos_tokens.reshape(-1, model_token_size, model_token_size, embedding_size).contiguous().permute(0, 3, 1, 2) # bs, c, h, w
            search_pos_tokens = torch.nn.functional.interpolate(
                model_pos_tokens_temp, size=(search_size, search_size), mode='bicubic', align_corners=False)
            search_pos_tokens = search_pos_tokens.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        else:
            search_pos_tokens = model_pos_tokens
        new_pos_embed = torch.cat((extra_pos_tokens, search_pos_tokens, search_pos_tokens), dim=1)
        new_pos_embed_three_frame = torch.cat((extra_pos_tokens, search_pos_tokens, search_pos_tokens, search_pos_tokens), dim=1)
        return new_pos_embed, new_pos_embed_three_frame


# borrowed from https://github.com/ViTAE-Transformer/ViTDet/blob/main/mmdet/models/backbones/vitae.py
class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, single_object=False, img_size=None,  **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            self.single_object = single_object

        self.mask_patch_embed = PatchEmbed(
                img_size=img_size, patch_size=16, in_chans=1, embed_dim=768) # !!! to check whether it has grads
        
        self.fpn1 = nn.Sequential(  # 1/4
            nn.ConvTranspose2d(768, 768, kernel_size=2, stride=2),
            Norm2d(768),
            nn.GELU(),
            nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(   # 1/8
            nn.ConvTranspose2d(768, 512, kernel_size=2, stride=2),
        )

        self.stcn_decoder = Decoder()

        self.temporal_pos_embed_x = nn.Parameter(torch.randn(1, 1, 768)) # for search, requires_grad=True
        self.temporal_pos_embed_z = nn.Parameter(torch.randn(1, 1, 768)) # for template, requires_grad=True


    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1-prob, dim=1, keepdim=True),  # get the background region
            prob
        ], 1).clamp(1e-7, 1-1e-7)
        logits = torch.log((new_prob /(1-new_prob))) # bs, 3, 384, 384
        return logits


    def forward(self, mode=None, **kwargs): #memory_frames=None, mask_frames=None,  query_frame=None, mode=None, selector=None):
        '''
        memory_frames: bs, T, 3, 384, 384
        mask_frames: bs, T, 3, 384, 384
        query_frame: bs, 3, 384, 384
        '''
        
        if mode == 'backbone':
            memory_frames = kwargs['memory_frames']
            mask_frames = kwargs['mask_frames']
            query_frame = kwargs['query_frame']
            B, T, C, H, W = memory_frames.shape
            
            memory_frames = memory_frames.flatten(0, 1)
            mask_frames = mask_frames.flatten(0, 1)
            memory_tokens = self.patch_embed(memory_frames)    # bs*T, (H//16 * W//16), 768
            mask_tokens   = self.mask_patch_embed(mask_frames) # bs*T, (H//16 * W//16), 768
            # add the target-aware positional encoding
            memory_tokens = memory_tokens + mask_tokens
            memory_tokens = memory_tokens + self.temporal_pos_embed_z

            query_tokens = self.patch_embed(query_frame) # bs, (H//16 * W//16), 768
            query_tokens = query_tokens + self.temporal_pos_embed_x

            if T > 1: # multiple memory frames
                memory_tokens = memory_tokens.view(B, T, -1, memory_tokens.size()[-1]).contiguous() #bs ,T, num, C
                # use all the memory frames
                memory_tokens = memory_tokens.flatten(1, 2) # bs , num, C
            
            x = torch.cat((memory_tokens, query_tokens), dim=1)
            if T > 1: # if there are multiple memory frames are used.
                single_size = int((self.pos_embed_two_frame[:, 1:, :].shape[1]) / 2)
                x = x + self.pos_embed_two_frame[:, 1:(single_size+1), :].repeat(1, T+1, 1)
            else:
                x = x + self.pos_embed_two_frame[:, 1:, :] # only using one memory frame, also including the current search frame
            x = self.pos_drop(x)
            for blk in self.blocks:
                x = blk(x)
            
            # maybe we need the norm(x), improves the results!
            x = self.norm(x)

            num_query_tokens = query_tokens.shape[1]
            updated_query_tokens = x[:, -num_query_tokens:, :]
            updated_query_tokens = updated_query_tokens.permute(0, 2, 1).contiguous().view(B, 768, int(H//16), int(W//16))
            m16 =  updated_query_tokens             # bs, 768, 24, 24
            m8  =  self.fpn2(updated_query_tokens)  # bs, 512, 48, 48
            m4  =  self.fpn1(updated_query_tokens)  # bs, 256, 96, 96

            return m16, m8, m4
        elif mode == 'segmentation':
            # print('decoder for segmentation')
            m16 = kwargs['m16']
            m8 = kwargs['m8']
            m4 = kwargs['m4']
            selector = kwargs['selector']

            # m16=m16, m8=m8, m4=m4, selector=selector
            if self.single_object:
                logits = self.stcn_decoder(m16, m8, m4)
                prob = torch.sigmoid(logits)
            else:
                
                # qf8: 4, 512, 48, 48; qf4: 4, 256, 96, 96;
                logits = torch.cat([
                    self.stcn_decoder(m16[:,0], m8[:,0], m4[:,0]), 
                    self.stcn_decoder(m16[:,1], m8[:,1], m4[:,1]), 
                ], 1)  # 4, 2, 384, 384

                prob = torch.sigmoid(logits) # 4, 2, 384, 384; 2: two targets
                prob = prob * selector.unsqueeze(2).unsqueeze(2) # 4, 2, 384, 384 

            logits = self.aggregate(prob)
            prob = F.softmax(logits, dim=1)[:, 1:]
            return logits, prob
        elif mode == 'segmentation_single_onject':
            m16 = kwargs['m16']
            m8 = kwargs['m8']
            m4 = kwargs['m4']
            logits = self.stcn_decoder(m16, m8, m4)
            return torch.sigmoid(logits)


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

    