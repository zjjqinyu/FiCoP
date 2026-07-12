import torch
from typing import OrderedDict, Tuple, List
from omegaconf import DictConfig
from torch import Tensor

from models.vlm import get_vlm
from models.fusion import get_fusion_module
from models.decoder import get_decoder
from models.dinov2 import get_dinov2
from models.cpgp import get_cpgp_module
from models.patch_corrs_head import get_patch_corrs_head
from torch import nn

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Upsample) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Net(torch.nn.Module):
    def __init__(self, args : DictConfig, device : str):
        super().__init__()
        self.args = args.model
        self.device = device
        self.vlm = get_vlm(self.args, self.device)
        self.dinov2 = get_dinov2(self.args, self.device)
        self.fusion = get_fusion_module(self.args, self.device)
        self.decoder = get_decoder(self.args, self.device)
        self.cpgp = get_cpgp_module(self.args, self.device)
        self.patch_corrs_head = get_patch_corrs_head(self.args, self.device)
        self.conv_transpose_lst = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=1),
            nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(1024, 128, kernel_size=4, stride=4)
            ])
        self.init_all()

    def get_trainable_parameters(self) -> list:
        param_list = []
        param_list.extend(self.fusion.parameters())
        param_list.extend(self.decoder.parameters())
        param_list.extend(self.cpgp.parameters())
        param_list.extend(self.patch_corrs_head.parameters())

        return param_list

    def get_guidance_embeds(self, feat_list : Tensor, in_index: List[int]) -> List[Tensor]:
        out_feats = []
        for i, in_idx in enumerate(in_index):
            out_feats.append(self.conv_transpose_lst[i](feat_list[in_idx]))
        return out_feats

    def train(self, mode=True):
        self.training = mode
        self.vlm.train(mode)
        self.fusion.train(mode)
        self.decoder.train(mode)
        self.cpgp.train(mode)
        self.patch_corrs_head.train(mode)
        
        return self

    def eval(self):
        self.train(False)

    def get_image_input(self, xs : dict) -> Tuple[dict, dict]:
        # create input with RGB channels
        input_a = {'rgb': xs['anchor']['rgb'].to(self.device)}
        input_q = {'rgb': xs['query']['rgb'].to(self.device)}

        return (input_a, input_q)

    def init_all(self):
        self.fusion.clip_conv.apply(weights_init_kaiming)
        
        if self.args.use_catseg_ckpt:
            #print("Loading CATSeg checkpoint")
            ckpt = torch.load('pretrained_models/catseg.pth', map_location=self.device)
            # set checkpoint names
            new_state_dict = dict()
            # this is necessary because of the refactoring we carried out
            old_fusion_key = 'sem_seg_head.predictor.transformer'
            new_fusion_key = 'fusion'
            old_dec_key = 'fusion.decoder'
            new_dec_key = 'decoder.decoder'
            
            # changing prefix of fusion and decoder keys
            for k,v in ckpt['model'].items():
                if k.startswith(old_fusion_key):
                    new_k = k.replace(old_fusion_key, new_fusion_key)
                    if new_k.startswith(old_dec_key):
                        new_k = new_k.replace(old_dec_key, new_dec_key)
                    # if new_k.startswith('fusion.head'):
                    #     new_k = new_k.replace('fusion.head', 'decoder.head')
                    new_state_dict[new_k] = v
                
            # if using CLIP, we are also loading CATSeg's finetuned CLIP    
            if self.args.image_encoder.vlm == 'clip':
                old_clip_key = 'sem_seg_head.predictor.clip_model'
                new_clip_key = 'vlm.clip_model'
            
                for k,v in ckpt['model'].items():
                    if k.startswith(old_clip_key):
                        new_k = k.replace(old_clip_key,new_clip_key)
                        new_state_dict[new_k] = v
                            
            inco_keys = self.load_state_dict(new_state_dict,strict=False)
            #print(inco_keys)

        else:
            #print("Training from scratch")            
            self.fusion.apply(weights_init_kaiming)
            self.decoder.apply(weights_init_kaiming)
            self.cpgp.apply(weights_init_kaiming)
            self.patch_corrs_head.apply(weights_init_kaiming)


    def forward(self, xs: dict):
        visual_a_feat_lst, visual_a = self.dinov2.extract_intermediate_features(xs['anchor']['rgb'])
        visual_q_feat_lst, visual_q = self.dinov2.extract_intermediate_features(xs['query']['rgb'])


        prompt_emb = self.vlm.encode_prompt(xs['prompt'])

        guid_a = self.get_guidance_embeds(visual_a_feat_lst, [23,15,7])
        guid_q = self.get_guidance_embeds(visual_q_feat_lst, [23,15,7])

        # get encoded feature maps [D,N,N]
        prompt_emb = prompt_emb.unsqueeze(1)
        feats_a = self.fusion.forward(visual_a, prompt_emb, guid_a)
        feats_q = self.fusion.forward(visual_q, prompt_emb, guid_q) # [32, 128, 1, 24, 24]

        feats_a, feats_q = self.cpgp.forward(feats_a.squeeze(2), feats_q.squeeze(2), prompt_emb.squeeze(1))
        patch_corrs_map = self.patch_corrs_head(feats_a, feats_q)

        feats_a = feats_a.unsqueeze(2)
        feats_q = feats_q.unsqueeze(2)

        featmap_a = self.decoder.forward(feats_a, guid_a)
        featmap_q = self.decoder.forward(feats_q, guid_q)
        
        assert featmap_a.shape[2:] == self.args.image_encoder.img_size
        
        return {
            'featmap_a' : featmap_a,
            'featmap_q' : featmap_q,
            'mask_a' : xs['anchor']['cropped_pred_mask'],
            'mask_q' : xs['query']['cropped_pred_mask'],
            'patch_corrs_map' : patch_corrs_map
        }
