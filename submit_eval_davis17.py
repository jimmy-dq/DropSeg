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
import cv2


from model import models_vit
import timm
assert timm.__version__ == "0.3.2" # version check


def overlay_davis(image,mask,colors=[255,0,0],cscale=2,alpha=0.4):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # import skimage
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours,:] = 0

    return im_overlay.astype(image.dtype)


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


# python evaluation_method.py --task semi-supervised --results_path /home/user/Projects/STCN/taiji_May16_13.43.44_retrain_s03_37500
"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='/home/user/Projects/STCN/saves/stcn_s03.pth')
parser.add_argument('--davis_path', default='/home/user/Data/DAVIS/2017')
parser.add_argument('--output')
parser.add_argument('--split', help='val/testdev', default='val') #val testdev
parser.add_argument('--amp', action='store_true')
args = parser.parse_args()

checkpoint_list = [210000]
checkpoint_name = 'Nov06_22.16.18_retrain_s03'
# evaluate one model with various epochs
for check_index in range(len(checkpoint_list)):
            
            model_e = checkpoint_list[check_index]

            model_name = checkpoint_name + '_' + str(model_e) + '.pth'
            print(model_name)
            args.output = model_name[0:-4]
            davis_path = args.davis_path
            out_path = args.output

            # Simple setup
            os.makedirs(out_path, exist_ok=True)
            palette = Image.open(path.expanduser(davis_path + '/trainval/Annotations/480p/blackswan/00000.png')).getpalette()

            torch.autograd.set_grad_enabled(False)

            # Setup Dataset
            if args.split == 'val':
                test_dataset = DAVISTestDataset(davis_path+'/trainval', imset='2017/val.txt')
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
            elif args.split == 'testdev':
                test_dataset = DAVISTestDataset(davis_path+'/test-dev', imset='2017/test-dev.txt')
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
            else:
                raise NotImplementedError

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
                    size = info['size_480p']

                    torch.cuda.synchronize()
                    process_begin = time.time()

                    print('before_interpolation:')
                    print(pos_embed_two_frame.shape)
                    processor = InferenceCore_ViT(vit_model, rgb, k, pos_embed_two_frame)
                    processor.interact(msk[:,0], 0, rgb.shape[1])

                    # Do unpad -> upsample to original size 
                    out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
                    for ti in range(processor.t):
                        prob = unpad(processor.prob[:,ti], processor.pad)
                        prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
                        out_masks[ti] = torch.argmax(prob, dim=0)
                    
                    out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

                    torch.cuda.synchronize()
                    total_process_time += time.time() - process_begin
                    total_frames += out_masks.shape[0]


                    # # # save to overlay images
                    # video_name =  data['info']['name'][0]
                    # base_path = os.path.join('/apdcephfs/private_qiangqwu/Projects/STCN_ft/qualitative_results', video_name)
                    # os.makedirs(base_path)
                    # for ti in range(processor.t):
                    #     pF = cv2.imread(os.path.join('/apdcephfs/share_1290939/qiangqwu/VOS/DAVIS/2017/trainval/JPEGImages/480p', video_name, data['info']['frames'][ti][0]))
                    #     pF = cv2.cvtColor(pF, cv2.COLOR_BGR2RGB)
                    #     canvas = overlay_davis(pF, out_masks[ti], palette)
                    #     canvas = Image.fromarray(canvas)
                    #     canvas.save(os.path.join(base_path, data['info']['frames'][ti][0]))

                    # Save the results
                    this_out_path = path.join(out_path, name)
                    os.makedirs(this_out_path, exist_ok=True)
                    for f in range(out_masks.shape[0]):
                        img_E = Image.fromarray(out_masks[f])
                        img_E.putpalette(palette)
                        img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))

                    del rgb
                    del msk
                    del processor

            print('Total processing time: ', total_process_time)
            print('Total processed frames: ', total_frames)
            print('FPS: ', total_frames / total_process_time)
