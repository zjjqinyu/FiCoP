import os

import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from utils.data import nocs, shapenet, toyl
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

from segment_anything import SamPredictor, sam_model_registry
from pycocotools import mask as maskUtils

import matplotlib.pyplot as plt

class OvBboxPredictor():
    def __init__(self, dataset_name=None, load_model=False, cache_dir='data/bbox_cache'):
        assert dataset_name in [None, 'shapenet6d', 'nocs', 'toyl']
        self.dataset_name = dataset_name
        self.cache_dict = None
        if dataset_name is not None:
            self.root = os.path.join('data', dataset_name)
            if dataset_name == 'shapenet6d':
                self.metadata = shapenet.get_metadata(self.root)
            elif dataset_name == 'nocs':
                self.obj_names = nocs.get_obj_names(self.root)
            elif dataset_name == 'toyl':
                self.obj_names = toyl.get_obj_names(self.root)
            else:
                raise ValueError
            cache_path = os.path.join(cache_dir, f'{dataset_name}.json')
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    self.cache_dict = json.load(f)

        self.load_model = load_model
        if load_model:
            self.model = self._load_grounding_model()

    def create_cache(self, output_cache_dir='data/bbox_cache'):
        assert self.dataset_name in ['shapenet6d', 'nocs', 'toyl']
        save_path = os.path.join(output_cache_dir, f'{self.dataset_name}.json')
        if not os.path.exists(output_cache_dir):
            os.mkdir(output_cache_dir)
        if self.dataset_name == 'shapenet6d':
            path_split = os.path.join(self.root, 'fixed_split', 'custom_train')
            with open(os.path.join(path_split,'instance_list.txt')) as f:
                instances = f.readlines()
            data = {}
            for instance in tqdm(instances):
                idx_a, idx_q, obj_id = instance.split(',')
                idx_a, idx_q, obj_id = int(idx_a), int(idx_q), int(obj_id)
                instance_a = f'{idx_a} {obj_id}'
                instance_q = f'{idx_q} {obj_id}'
                bbox_info_a = self.get_bbox_info_from_model(instance_a)
                bbox_info_q = self.get_bbox_info_from_model(instance_q)
                data[instance_a] = bbox_info_a
                data[instance_q] = bbox_info_q
            with open(save_path, 'w') as f:
                json.dump(data, f)
        elif self.dataset_name == 'nocs':
            path_split = os.path.join(self.root, 'fixed_split', 'cross_scene_test')
            with open(os.path.join(path_split,'instance_list.txt')) as f:
                instances = f.readlines()
            data = {}
            for instance in tqdm(instances):
                split, idx_a, idx_q, cat_id = instance.split(',')
                cat_id_a, obj_name_a = cat_id.strip().split(' ')
                cat_id_a = int(cat_id_a)
                scene_a, img_a = [int(n) for n in idx_a.split(' ') if n != '']
                scene_q, img_q = [int(n) for n in idx_q.split(' ') if n != '']

                instance_a = f'{scene_a} {img_a} {obj_name_a}'
                instance_q = f'{scene_q} {img_q} {obj_name_a}'
                bbox_info_a = self.get_bbox_info_from_model(instance_a)
                bbox_info_q = self.get_bbox_info_from_model(instance_q)
                data[instance_a] = bbox_info_a
                data[instance_q] = bbox_info_q
            with open(save_path, 'w') as f:
                json.dump(data, f)
        elif self.dataset_name == 'toyl':
            path_split = os.path.join(self.root, 'fixed_split', 'cross_scene_test')
            with open(os.path.join(path_split,'instance_list.txt')) as f:
                instances = f.readlines()
            data = {}
            for instance in tqdm(instances):
                split, idx_a, idx_q, cls_id = instance.split(',')
                cls_id = int(cls_id)
                scene_a, img_a = [int(n) for n in idx_a.split(' ') if n != '']
                scene_q, img_q = [int(n) for n in idx_q.split(' ') if n != '']

                instance_a = f'{scene_a} {img_a} {cls_id}'
                instance_q = f'{scene_q} {img_q} {cls_id}'
                bbox_info_a = self.get_bbox_info_from_model(instance_a)
                bbox_info_q = self.get_bbox_info_from_model(instance_q)
                data[instance_a] = bbox_info_a
                data[instance_q] = bbox_info_q
            with open(save_path, 'w') as f:
                json.dump(data, f)
        else:
            raise ValueError

    def visualize_from_cache(self, instance_id, save_path):
        bbox_info = self.get_bbox_info_from_cache(instance_id)
        self.visualize_from_bbox_info(bbox_info, save_path)

    def visualize_from_model(self, instance_id, save_path):
        bbox_info = self.get_bbox_info_from_model(instance_id)
        self.visualize_from_bbox_info(bbox_info, save_path)

    def visualize_from_bbox_info(self, bbox_info, save_path):
        if bbox_info is not None:
            image_pil = Image.open(bbox_info['img_path']).convert("RGB")
            tgt = {}
            tgt["size"] = image_pil.size[1], image_pil.size[0]
            tgt["boxes"] = torch.tensor([bbox_info['bbox']])
            tgt["labels"] = [bbox_info['prompt']]
            res_img = self.plot_boxes_to_image(image_pil, tgt)[0]
            res_img.save(save_path)

    def get_bbox_info(self, instance_id):
        if self.cache_dict is not None:
            return self.get_bbox_info_from_cache(instance_id)
        elif self.load_model:
            return self.get_bbox_info_from_model(instance_id)
        else:
            raise NotImplementedError

    def get_bbox_info_from_cache(self, instance_id):
        assert self.cache_dict is not None
        bbox_info = self.cache_dict[instance_id]
        return bbox_info
    
    def get_bbox_info_from_model(self, instance_id):
        image_path, text_prompt = self._get_image_path_and_prompt_from_instance_id(instance_id)
        bbox = self.get_bbox_from_image_and_prompt(image_path, text_prompt)
        return {
            'bbox': bbox,
            'img_path': image_path,
            'prompt': text_prompt
        }

    def get_bbox_from_image_and_prompt(self, image_path, text_prompt):
        assert self.load_model
        _, image = self._load_image(image_path)
        boxes_filt, _ = self._get_grounding_output(
            image, text_prompt, box_threshold=0.1, text_threshold=0.25, cpu_only=False, token_spans=None)
        if len(boxes_filt) == 0:
            bbox = [0., 0., 0., 0.]
        else:
            bbox = boxes_filt[0].tolist()
        return bbox     # xywh(norm)
    
    def plot_boxes_to_image(self, image_pil, tgt):
        H, W = tgt["size"]
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        assert len(boxes) == len(labels), "boxes and labels must have same length"

        draw = ImageDraw.Draw(image_pil)
        mask = Image.new("L", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        # draw boxes and masks
        for box, label in zip(boxes, labels):
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            # draw
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
            # draw.text((x0, y0), str(label), fill=color)

            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), str(label), font)
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (x0, y0, w + x0, y0 + h)
            # bbox = draw.textbbox((x0, y0), str(label))
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), str(label), fill="white")

            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

        return image_pil, mask
    
    def _get_image_path_and_prompt_from_instance_id(self, instance_id):
        assert self.dataset_name in ['shapenet6d', 'nocs', 'toyl']
        if self.dataset_name == 'shapenet6d':
            img_id, cat_id = instance_id.split()
            img_id, cat_id = int(img_id), int(cat_id)
            image_path = os.path.join(self.root,'raw_data','rgb',f'{img_id:06d}.jpg')

            cat_map, id_new2old, _ = self.metadata
            text = cat_map[id_new2old[cat_id]]['obj_syn']
            text_prompt = text[0]

        elif self.dataset_name == 'nocs':
            scene_id, img_id, obj_name = instance_id.split()
            scene_id, img_id = int(scene_id), int(img_id)
            base_path = os.path.join(self.root, 'split/real_test', f'scene_{scene_id}/{img_id:04d}') 
            image_path = base_path + '_color.png'

            text = self.obj_names[obj_name]
            text_prompt = f'{text[1]} {text[0]}'
            # text_prompt = f'{text[2]} {text[0]}' # wrong text

        elif self.dataset_name == 'toyl':
            scene_id, img_id, cls_id = instance_id.split()
            scene_id, img_id = int(scene_id), int(img_id)
            base_path = os.path.join(self.root, 'split', 'test', f'{scene_id:06d}') 
            image_path = os.path.join(base_path,'rgb',f'{img_id:06d}' + '.png')

            text = self.obj_names[cls_id]
            text_prompt = f'{text[1]} {text[0]}'
            # text_prompt = f'{text[2]} {text[0]}' # wrong text

        else:
            raise ValueError
        return image_path, text_prompt
    
    def _load_image(self, image_path):
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image
    
    def _load_grounding_model(self, model_checkpoint_path='pretrained_models/groundingdino/groundingdino_swinb_cogcoor.pth', cpu_only=False):
        import pkg_resources
        model_config_path = pkg_resources.resource_filename(
            'groundingdino.config', 
            'GroundingDINO_SwinB_cfg.py'
        )
        args = SLConfig.fromfile(model_config_path)
        args.device = "cuda" if not cpu_only else "cpu"
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model
    def _get_grounding_output(self, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
        assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
        model = self.model
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = "cuda" if not cpu_only else "cpu"
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # filter output
        if token_spans is None:
            logits_filt = logits.cpu().clone()
            boxes_filt = boxes.cpu().clone()
            filt_mask = logits_filt.max(dim=1)[0] > box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

            # get phrase
            tokenlizer = model.tokenizer
            tokenized = tokenlizer(caption)
            # build pred
            pred_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
                if with_logits:
                    pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                else:
                    pred_phrases.append(pred_phrase)
        else:
            # given-phrase mode
            positive_maps = create_positive_map_from_span(
                model.tokenizer(text_prompt),
                token_span=token_spans
            ).to(image.device) # n_phrase, 256

            logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
            all_logits = []
            all_phrases = []
            all_boxes = []
            for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
                # get phrase
                phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
                # get mask
                filt_mask = logit_phr > box_threshold
                # filt box
                all_boxes.append(boxes[filt_mask])
                # filt logits
                all_logits.append(logit_phr[filt_mask])
                if with_logits:
                    logit_phr_num = logit_phr[filt_mask]
                    all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
                else:
                    all_phrases.extend([phrase for _ in range(len(filt_mask))])
            boxes_filt = torch.cat(all_boxes, dim=0).cpu()
            pred_phrases = all_phrases

        return boxes_filt, pred_phrases
        
class OvMaskPredictor():
    def __init__(self, dataset_name=None, load_det_model=False, load_sam_model=False, mask_cache_dir='data/mask_cache', bbox_cache_dir='data/bbox_cache'):
        assert dataset_name in [None, 'shapenet6d', 'nocs', 'toyl']
        self.dataset_name = dataset_name
        self.bbox_predictor = OvBboxPredictor(dataset_name, load_det_model, bbox_cache_dir)
        self.load_sam_model = load_sam_model
        self.cache_dict = None
        if dataset_name is not None:
            cache_path = os.path.join(mask_cache_dir, f'{dataset_name}.json')
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    self.cache_dict = json.load(f)
                for mask_info in self.cache_dict.values():
                    rle = mask_info.pop('rle')
                    rle["counts"] = rle["counts"].encode("utf-8")
                    mask = maskUtils.decode(rle)
                    mask_info['mask'] = mask
        if load_sam_model:
            sam = sam_model_registry["vit_h"](checkpoint="pretrained_models/sam/sam_vit_h_4b8939.pth")
            self.predictor = SamPredictor(sam.cuda())
        
    def create_cache(self, output_cache_dir='data/mask_cache'):
        assert self.bbox_predictor.cache_dict is not None
        bbox_cache_dict = self.bbox_predictor.cache_dict
        if not os.path.exists(output_cache_dir):
            os.mkdir(output_cache_dir)
        save_path = os.path.join(output_cache_dir, f'{self.bbox_predictor.dataset_name}.json')
        data = {}
        for instance_id, bbox_info in tqdm(bbox_cache_dict.items()):
            mask_info = self.get_mask_from_model(instance_id)
            mask = mask_info.pop('mask')
            rle =  maskUtils.encode(np.asarray(mask, order='F'))
            rle["counts"] = rle["counts"].decode("utf-8")
            mask_info['rle'] = rle
            data[instance_id] = mask_info
        with open(save_path, 'w') as f:
            json.dump(data, f)

    def get_mask_from_image_and_bbox(self, img_path, bbox): # bbox format: norm(xywh)
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        bbox = bbox * np.array([W, H, W, H]) # norm ->
        # from xywh to xyxy
        bbox[:2] -= bbox[2:] / 2
        bbox[2:] += bbox[:2]
        self.predictor.set_image(np.array(img))
        masks, _, _ = self.predictor.predict(box=bbox, multimask_output=False)
        return masks[0]
    
    def get_mask_from_image_and_prompt(self, image_path, text_prompt):
        assert self.load_sam_model and self.bbox_predictor.load_model
        bbox = self.bbox_predictor.get_bbox_from_image_and_prompt(image_path, text_prompt)
        bbox = np.array(bbox)
        mask = self.get_mask_from_image_and_bbox(image_path, bbox)
        return mask

    def get_mask_from_model(self, instance_id):
        assert self.load_sam_model
        bbox_info = self.bbox_predictor.get_bbox_info(instance_id)
        img_path = bbox_info['img_path']
        bbox = np.array(bbox_info['bbox'])
        mask = self.get_mask_from_image_and_bbox(img_path, bbox)
        mask_info = {
            'mask': mask,
            'img_path': bbox_info['img_path'],
            'prompt': bbox_info['prompt']
        }
        return mask_info

    def get_mask_info_from_cache(self, instance_id):
        assert self.cache_dict is not None
        mask_info = self.cache_dict[instance_id]
        return mask_info
    
    def plot_mask_to_image(self, image_path, mask, save_path, alpha=0.5):
        image = np.array(Image.open(image_path).convert("RGB"))
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.imshow(mask, cmap='jet', alpha=alpha)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    bbox_predictor = OvBboxPredictor(dataset_name='nocs', load_model=True)
    bbox_predictor.create_cache(output_cache_dir='data/bbox_cache')

    mask_predictor = OvMaskPredictor('nocs', load_det_model=False, load_sam_model=True)
    mask_predictor.create_cache(output_cache_dir='data/mask_cache')

    bbox_predictor = OvBboxPredictor(dataset_name='toyl', load_model=True)
    bbox_predictor.create_cache(output_cache_dir='data/bbox_cache')

    mask_predictor = OvMaskPredictor('toyl', load_det_model=False, load_sam_model=True)
    mask_predictor.create_cache(output_cache_dir='data/mask_cache')

    # mask_predictor = OvMaskPredictor('nocs', load_det_model=False, load_sam_model=False)
    # mask_info = mask_predictor.get_mask_info_from_cache('4 456 camera_canon_len_norm')
    # mask_predictor.plot_mask_to_image(mask_info['img_path'], mask_info['mask'], 'ov_img.png')


    