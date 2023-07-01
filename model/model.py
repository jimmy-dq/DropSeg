"""
model.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""


import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from model.network import STCN
from model.losses import LossComputer, iou_hooks_mo, iou_hooks_so
from util.log_integrator import Integrator
from util.image_saver import pool_pairs

from model import models_vit
import timm
assert timm.__version__ == "0.3.2" # version check

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

import numpy as np
from typing import Tuple
# patch embeding for CLIP
class PatchEmbed2D(nn.Module):

    def __init__(
        self,
        patch_size: Tuple[int, int] = (16, 16),
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels

        self.proj = nn.Linear(np.prod(patch_size) * in_channels, embed_dim)


    def _initialize_weights(self, x):
        nn.init.kaiming_normal_(self.proj.weight, 0.)
        nn.init.constant_(self.proj.bias, 0.)


    def forward(self, x: torch.Tensor, is_template=True):
        B, C, H, W = x.size()
        pH, pW = self.patch_size

        assert C == self.in_channels and H % pH == 0 and W % pW == 0

        x = x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 1, 3, 5).flatten(3).flatten(1, 2)
        x = self.proj(x)
        
        return x

class ViTSTCNModel:
    def __init__(self, para, logger=None, save_path=None, local_rank=0, world_size=1):
        # we use the ViT model to perform the joint feature exctraction and interaction
        self.para = para
        self.single_object = para['single_object']   # default: False, multiple objects per frame;
        vit_model = models_vit.__dict__['vit_base_patch16'](
                num_classes=1000,
                drop_path_rate=para['droppath_rate'], #0.1
                global_pool=True,
                single_object = self.single_object,
                img_size=para['img_size'])
        print(self.para)

        print('load %s weights!!!' %self.para['pretrained_net_path'])
        if self.para['pretrained_net_path'] == '':
            print('no pretraining model is loaded')
        else:
            checkpoint = torch.load(self.para['pretrained_net_path'], map_location='cpu')
            print(checkpoint.keys())
            # print(checkpoint)
            checkpoint_model = checkpoint['model']
        
            key_names = list(checkpoint_model.keys())
            for key_name in key_names:
                if 'decoder' in key_name:
                    del checkpoint_model[key_name]
            state_dict = vit_model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            # load pre-trained model
            msg = vit_model.load_state_dict(checkpoint_model, strict=False)
            print(msg)



        # interpolate the position embedding
        print('interpolation after the pretrained weight loaded!!!!!!')
        pos_embed_two_frame, pos_embed_three_frame = interpolate_pos_embed(vit_model.pos_embed, int(para['img_size']//16))
        vit_model.pos_embed_two_frame, vit_model.pos_embed_three_frame = torch.nn.Parameter(pos_embed_two_frame, requires_grad=False), torch.nn.Parameter(pos_embed_three_frame, requires_grad=False)
        print(vit_model.pos_embed_two_frame)
        print(vit_model.pos_embed_three_frame)

        ################### Original Setting used in STCN ##############################
        
        self.local_rank = local_rank

        self.STCN = nn.parallel.DistributedDataParallel(
            vit_model.cuda(), 
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)

        # Setup logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        if self.single_object:
            self.train_integrator.add_hook(iou_hooks_so)
        else:
            self.train_integrator.add_hook(iou_hooks_mo)
        self.loss_computer = LossComputer(para)

        self.train()
        self.optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.STCN.parameters()), lr=para['lr'], weight_decay=para['weight_decay'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, para['steps'], para['gamma'])


        if para['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.report_interval = 100
        self.save_im_interval = 2000
        self.save_model_interval = 10000 #50000
        self.img_size = para['img_size']
        if para['debug']:
            self.report_interval = self.save_im_interval = 1
    
    
    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        Fs = data['rgb'] # bs, 3, 3, 384, 384
        Ms = data['gt']  # bs, 3, 1, 384, 384

        with torch.cuda.amp.autocast(enabled=self.para['amp']):
            # key features never change, compute once
            # k16: key: 64-D, kf16: key:512-D; kf16, kf8, kf4, which are all output from the backbone

            if self.single_object:
                m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.STCN(memory_frames=Fs[:,0].unsqueeze(1), mask_frames=Ms[:,0].unsqueeze(1),  query_frame=Fs[:,1], mode='backbone')
                prev_logits, prev_mask = self.STCN(m16=m16_f1_v1, m8=m8_f1_v1,  m4=m4_f1_v1, selector=torch.ones((Fs.shape[0], 1)).cuda(), mode='segmentation')

                out['mask_1'] = prev_mask
                # out['mask_2'] = this_mask
                out['logits_1'] = prev_logits
                # out['logits_2'] = this_logits
            else:
                sec_Ms = data['sec_gt']         # 4, 3, 1, 384, 384
                selector = data['selector']     # 4, 2

                m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.STCN(memory_frames=Fs[:,0].unsqueeze(1), mask_frames=Ms[:,0].unsqueeze(1),  query_frame=Fs[:,1], mode='backbone')
                m16_f1_v2, m8_f1_v2, m4_f1_v2 = self.STCN(memory_frames=Fs[:,0].unsqueeze(1), mask_frames=sec_Ms[:,0].unsqueeze(1),  query_frame=Fs[:,1], mode='backbone')
                
                # Segment frame 1 with frame 0
                prev_logits, prev_mask = self.STCN(m16=torch.cat((m16_f1_v1.unsqueeze(1), m16_f1_v2.unsqueeze(1)), dim=1),
                           m8=torch.cat((m8_f1_v1.unsqueeze(1),   m8_f1_v2.unsqueeze(1)), dim=1), 
                           m4=torch.cat((m4_f1_v1.unsqueeze(1),   m4_f1_v2.unsqueeze(1)), dim=1), selector=selector, mode='segmentation')
                
                
                out['mask_1'] = prev_mask[:,0:1]      # frame-1-taregt-1
                out['sec_mask_1'] = prev_mask[:,1:2]  # frame-1-target-2

                out['logits_1'] = prev_logits          # frame-1 

            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, it)

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train:
                        if it % self.save_im_interval == 0 and it != 0:  #save_im_interval = 800
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (self.img_size, self.img_size)
                                self.logger.log_cv2('train/pairs', pool_pairs(images, size, self.single_object), it)

            if self._is_train:
                if (it) % self.report_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_model_interval == 0 and it != 0:    # self.save_model_interval = 50000
                    if self.logger is not None:
                        self.save(it)

            # Backward pass
            # This should be done outside autocast
            # but I trained it like this and it worked fine
            # so I am keeping it this way for reference
            self.optimizer.zero_grad(set_to_none=True)
            if self.para['amp']:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward() 
                self.optimizer.step()
            self.scheduler.step()

    def save(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % it)
        torch.save(self.STCN.module.state_dict(), model_path)
        print('Model saved to %s.' % model_path)

        self.save_checkpoint(it)

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = { 
            'it': it,
            'network': self.STCN.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.STCN.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it

    def load_network(self, path): #backbone_only
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        # key_names = list(src_dict.keys())

        # # here we only use the backbone, do not include the mask head and the multi-scale fpn 
        # if backbone_only:
        #     for k in key_names:
        #         if 'fpn' in k or 'stcn_decoder' in k:
        #             print('del ' + k)
        #             del src_dict[k]
        # # here we use the transformer backbone with the multi-scale fpn
        # else:
        #     for k in key_names:
        #         if 'stcn_decoder' in k:
        #             print('del ' + k)
        #             del src_dict[k]


        

        # # Maps SO weight (without other_mask) to MO weight (with other_mask)
        # for k in list(src_dict.keys()):
        #     if k == 'value_encoder.conv1.weight':
        #         if src_dict[k].shape[1] == 4:
        #             pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
        #             nn.init.orthogonal_(pads)
        #             src_dict[k] = torch.cat([src_dict[k], pads], 1)

        msc = self.STCN.module.load_state_dict(src_dict, strict=False)
        print(msc)
        print('Network weight loaded:', path)
        print('check grads for pos embed:')
        print(self.STCN.module.pos_embed_two_frame.requires_grad)
        print(self.STCN.module.pos_embed_three_frame.requires_grad)
        

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # Shall be in eval() mode to freeze BN parameters
        self.STCN.train()
        return self
        
    def val(self):
        self._is_train = False
        self._do_log = True
        self.STCN.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.STCN.eval()
        return self