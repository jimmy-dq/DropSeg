import os
from os import path
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from dataset.davis_test_dataset import DAVISTestDataset
from util.tensor_util import unpad
from inference_core import InferenceCore_ViT

from progressbar import progressbar
import torch.nn as nn


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

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='/home/user/Projects/STCN/saves/stcn_s03.pth')
parser.add_argument('--davis_path', default='/home/user/Data/DAVIS/2016')
parser.add_argument('--output')
parser.add_argument('--split', help='val/testdev', default='val') #val testdev
parser.add_argument('--amp', action='store_true')
args = parser.parse_args()

checkpoint_name = 'Nov06_22.16.18_retrain_s03' #'Oct31_00.49.53_retrain_s03' #'Oct31_00.08.09_retrain_s03' #'Oct28_01.29.26_retrain_s03' #'Oct26_21.09.25_retrain_s03' #'Oct26_21.32.27_retrain_s03' #'Oct26_21.09.25_retrain_s03' #'Oct26_21.05.31_retrain_s03' #'Oct22_00.43.02_retrain_s03' #'Oct12_17.02.16_retrain_s03' #'Oct24_13.57.39_retrain_s03' #'Oct23_04.35.41_retrain_s03' #'Oct22_00.43.02_retrain_s03' #'Oct18_02.44.07_retrain_s03' #'Oct15_12.47.25_retrain_s03' #'Oct11_13.32.29_retrain_s03' #'Oct11_14.38.12_retrain_s03'
checkpoint_list = [210000]
for check_index in range(len(checkpoint_list)):
            model_e = checkpoint_list[check_index]
            model_name = checkpoint_name + '_' + str(model_e) + '.pth'
            print(model_name)
            args.output = model_name[0:-4]  #+ '_refined_last_frame'
            davis_path = args.davis_path
            out_path = args.output

            # Simple setup
            os.makedirs(out_path, exist_ok=True)

            torch.autograd.set_grad_enabled(False)

            # Setup Dataset, a small hack to use the image set in the 2017 folder because the 2016 one is of a different format
            test_dataset = DAVISTestDataset(davis_path, imset='../../2017/trainval/ImageSets/2016/val.txt', single_object=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

            # Load our checkpoint
            vit_model = models_vit.__dict__['vit_base_patch16'](
                                num_classes=1000,
                                drop_path_rate=0.0, #0.1,
                                global_pool=True,
                                single_object = False,
                                img_size = 384) # 384 is the same as the training size


            pos_embed_two_frame, pos_embed_three_frame = interpolate_pos_embed(vit_model.pos_embed, int(384//16)) #384 is for training size
            vit_model.pos_embed_two_frame, vit_model.pos_embed_three_frame = torch.nn.Parameter(pos_embed_two_frame, requires_grad=False), torch.nn.Parameter(pos_embed_three_frame, requires_grad=False)
            checkpoint = torch.load(os.path.join('./checkpoints', model_name), map_location='cpu')
            print("Load pre-trained checkpoint from")
            state_dict = checkpoint

            # load pre-trained model
            msg = vit_model.load_state_dict(state_dict, strict=True)
            print(msg)
            vit_model = vit_model.cuda().eval()



            total_process_time = 0
            total_frames = 0

            pos_embed_two_frame = vit_model.pos_embed_two_frame.detach().clone()

            # Start eval
            for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

                with torch.cuda.amp.autocast(enabled=args.amp):   # per sequence here 
                    rgb = data['rgb'].cuda()    # 1, 69, 3, 480, 910 
                    msk = data['gt'][0].cuda()  # 2, 69, 1, 480, 910
                    info = data['info']         
                    name = info['name'][0]
                    k = len(info['labels'][0]) # num. of objects
                    # size = info['size_480p']

                    torch.cuda.synchronize()
                    process_begin = time.time()

                    print('before_interpolation:')
                    print(pos_embed_two_frame.shape)
                    processor = InferenceCore_ViT(vit_model, rgb, k, pos_embed_two_frame)
                    processor.interact(msk[:,0], 0, rgb.shape[1])

                    # Do unpad -> upsample to original size 
                    out_masks = torch.zeros((processor.t, 1, *rgb.shape[-2:]), dtype=torch.float32, device='cuda')
                    for ti in range(processor.t):
                        prob = processor.prob[:,ti]

                        if processor.pad[2]+processor.pad[3] > 0:
                            prob = prob[:,:,processor.pad[2]:-processor.pad[3],:]
                        if processor.pad[0]+processor.pad[1] > 0:
                            prob = prob[:,:,:,processor.pad[0]:-processor.pad[1]]

                        out_masks[ti] = torch.argmax(prob, dim=0)*255
                    
                    out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

                    torch.cuda.synchronize()
                    total_process_time += time.time() - process_begin
                    total_frames += out_masks.shape[0]

                    this_out_path = path.join(out_path, name)
                    os.makedirs(this_out_path, exist_ok=True)
                    for f in range(out_masks.shape[0]):
                        img_E = Image.fromarray(out_masks[f])
                        img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))

                    del rgb
                    del msk
                    del processor

            print('Total processing time: ', total_process_time)
            print('Total processed frames: ', total_frames)
            print('FPS: ', total_frames / total_process_time)