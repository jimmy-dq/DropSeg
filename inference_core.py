import torch

from model.eval_network import STCN
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by

from model import models_vit
import timm
assert timm.__version__ == "0.3.2" # version check



def interpolate_pos_embed_2D(pos_embed, kh, kw):
    
        num_extra_tokens = 1
        model_pos_tokens = pos_embed[:, num_extra_tokens:, :] 
        model_token_size = int((model_pos_tokens.shape[1]//2)**0.5)
        # pos_embed = net.pos_embed
        model_pos_tokens = pos_embed[:, num_extra_tokens:(model_token_size*model_token_size + 1), :] # bs, N, C
        extra_pos_tokens = pos_embed[:, :num_extra_tokens]

        embedding_size = extra_pos_tokens.shape[-1]

        if kh != model_token_size or kw != model_token_size: # do interpolation
            model_pos_tokens_temp = model_pos_tokens.reshape(-1, model_token_size, model_token_size, embedding_size).contiguous().permute(0, 3, 1, 2) # bs, c, h, w
            search_pos_tokens = torch.nn.functional.interpolate(
                model_pos_tokens_temp, size=(kh, kw), mode='bicubic', align_corners=False)
            search_pos_tokens = search_pos_tokens.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        else:
            search_pos_tokens = model_pos_tokens
        new_pos_embed = torch.cat((extra_pos_tokens, search_pos_tokens, search_pos_tokens), dim=1)
        new_pos_embed_three_frame = torch.cat((extra_pos_tokens, search_pos_tokens, search_pos_tokens, search_pos_tokens), dim=1)
        return new_pos_embed, new_pos_embed_three_frame


class InferenceCore_ViT:
    def __init__(self, prop_net:models_vit, images, num_objects, pos_embed_two_frame):
        self.prop_net = prop_net
        # self.mem_every = mem_every
        # self.include_last = include_last

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]

        # Pad each side to multiple of 16
        images, self.pad = pad_divide_by(images, 16)
        # Padded dimensions
        nh, nw = images.shape[-2:]

        self.images = images
        self.device = 'cuda'

        self.k = num_objects

        # Background included, not always consistent (i.e. sum up to 1)
        self.prob = torch.zeros((self.k+1, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob[0] = 1e-7   # for the background 

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16


        pos_embed_two_frame_new, pos_embed_three_frame_new  = interpolate_pos_embed_2D(pos_embed_two_frame, self.kh, self.kw)  # 75.1 variant
        self.prop_net.pos_embed_two_frame = torch.nn.Parameter(pos_embed_two_frame_new)
        self.prop_net.pos_embed_three_frame = torch.nn.Parameter(pos_embed_three_frame_new)

        print('after interpolation:')
        print(self.prop_net.pos_embed_two_frame.shape)

        print('init inference_core')

    def encode_key(self, idx):
        result = self.prop_net.encode_key(self.images[:,idx].cuda())
        return result

    def do_pass(self, target_initial_mask, idx, end_idx):
        # self.mem_bank.add_memory(key_k, key_v)
        closest_ti = end_idx

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        end = closest_ti - 1

        for ti in this_range:
            mask_list = []
            for obj_index in range(self.k): # for different objects. here we can also use previous frames
                target_mask = target_initial_mask[obj_index].unsqueeze(0).unsqueeze(0) # 1,1,1, H, W
                m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=self.images[:,idx].unsqueeze(1), mask_frames=target_mask,  query_frame=self.images[:,ti], mode='backbone')
                    
                out_mask = self.prop_net(m16=m16_f1_v1, m8 = m8_f1_v1, m4 = m4_f1_v1, mode='segmentation_single_onject')
                mask_list.append(out_mask)
            out_mask = torch.stack(mask_list, dim=0).flatten(0, 1) #num, 1, 1, h, w
            # do an concat here
            
            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[:,ti] = out_mask

        return closest_ti

    def interact(self, mask, frame_idx, end_idx):
        mask, _ = pad_divide_by(mask.cuda(), 16) # 2, 1, 480, 912

        self.prob[:, frame_idx] = aggregate(mask, keep_bg=True) # the 1st frame

        # Propagate
        self.do_pass(mask, frame_idx, end_idx)
