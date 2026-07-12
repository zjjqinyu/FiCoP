import torch

from os.path import join
from utils.data import nocs, shapenet, common, toyl

from datasets import CollateWrapper, Shapenet6DDataset, NOCSDataset, TOYLDataset, sample_correspondences, get_mask_type
from ov_models import OvMaskPredictor

def crop_item_img_via_pred_box(item, mask_predictor, eval=False):
    item['orig_mask'] = item['mask'].clone()
    item['orig_boxes'] = item['metadata']['boxes'].clone()

    orig_rgb = item['orig_rgb'].clone()
    orig_depth = item['orig_depth'].clone()
    C, H, W = orig_rgb.shape

    if eval:
        box = mask_predictor.bbox_predictor.get_bbox_info_from_cache(item['instance_id'])['bbox']
        box= torch.tensor(box).float()   # xywh(norm)
        box = box * torch.Tensor([W, H, W, H]).float()  # xywh
        x, y, w, h = box
        box = torch.tensor([y-h/2, x-w/2, h, w], dtype=torch.float32)   # y1x1hw
        pred_mask = mask_predictor.get_mask_info_from_cache(item['instance_id'])['mask']
        pred_mask = torch.tensor(pred_mask).to(torch.uint8)
    else:   # Using gt box and mask in training mode
        box = item['orig_boxes'].clone()    # y1x1hw
        pred_mask = item['orig_mask'].clone().to(torch.uint8)

    if torch.all(box == 0):
        box = torch.tensor([0, 0, H, W], dtype=torch.float32)   # y1x1hw
        pred_mask = torch.ones((H, W), dtype=torch.uint8)
    
    y1, x1, h, w, = box
    y2, x2 = y1 + h, x1 + w
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    cropped_rgb = orig_rgb[:, y1:y2, x1:x2]
    cropped_depth = orig_depth[y1:y2, x1:x2]
    cropped_mask = item['orig_mask'][y1:y2, x1:x2]
    cropped_pred_mask = pred_mask[y1:y2, x1:x2]

    item['rgb'] = cropped_rgb
    item['depth'] = cropped_depth.to(torch.int32)
    item['mask'] = cropped_mask.to(torch.uint8)
    item['pred_boxes'] = torch.tensor([y1, x1, y2-y1, x2-x1], dtype=torch.float32)    # xywh
    item['pred_mask'] = pred_mask.to(torch.uint8)
    item['cropped_pred_mask'] = cropped_pred_mask.to(torch.uint8)
    item['hw_size'] = item['mask'].shape
    item['metadata']['boxes'] = torch.tensor([0, 0, cropped_rgb.shape[0], cropped_rgb.shape[1]], dtype=torch.float32)
    return item

def get_new_corrs_after_crop(item_a, item_q, orig_corrs):
    corrs = orig_corrs.clone()

    bbox_a = item_a['pred_boxes'].clone()
    y1_a, x1_a, h_a, w_a = bbox_a
    a_x_min, a_y_min, a_x_max, a_y_max = x1_a, y1_a, x1_a+w_a, y1_a+h_a

    bbox_q = item_q['pred_boxes'].clone()
    y1_q, x1_q, h_q, w_q = bbox_q
    q_x_min, q_y_min, q_x_max, q_y_max = x1_q, y1_q, x1_q+w_q, y1_q+h_q
    
    in_bbox_a = (corrs[:, 1] >= a_x_min) & (corrs[:, 1] <= a_x_max) & \
                (corrs[:, 0] >= a_y_min) & (corrs[:, 0] <= a_y_max)
    

    in_bbox_q = (corrs[:, 3] >= q_x_min) & (corrs[:, 3] <= q_x_max) & \
                (corrs[:, 2] >= q_y_min) & (corrs[:, 2] <= q_y_max)
    
    valid_mask = in_bbox_a & in_bbox_q
    corrs = corrs[valid_mask].clone()
    corrs[:, 0] = corrs[:, 0] - y1_a
    corrs[:, 1] = corrs[:, 1] - x1_a  
    corrs[:, 2] = corrs[:, 2] - y1_q
    corrs[:, 3] = corrs[:, 3] - x1_q
    new_corrs = corrs
    # new_corrs = corrs - torch.tensor([y1_a, x1_a, y1_q, x1_q], device=corrs.device, dtype=corrs.dtype)
    return new_corrs

def compute_patch_corrs_map(img_size, corrs, grid_size):
    H, W = img_size
    patch_h = H // grid_size
    patch_w = W // grid_size
    
    y1_coords = corrs[:, 0]
    x1_coords = corrs[:, 1]
    y2_coords = corrs[:, 2]
    x2_coords = corrs[:, 3]
    
    patch_y_a = torch.div(y1_coords, patch_h, rounding_mode='trunc').long()
    patch_x_a = torch.div(x1_coords, patch_w, rounding_mode='trunc').long()
    patch_idx_a = patch_y_a * grid_size + patch_x_a
    
    patch_y_q = torch.div(y2_coords, patch_h, rounding_mode='trunc').long()
    patch_x_q = torch.div(x2_coords, patch_w, rounding_mode='trunc').long()
    patch_idx_q = patch_y_q * grid_size + patch_x_q
    
    N = grid_size * grid_size
    valid_mask = ((patch_idx_a >= 0) & (patch_idx_a < N) &
                 (patch_idx_q >= 0) & (patch_idx_q < N))
    
    valid_idx_a = patch_idx_a[valid_mask]
    valid_idx_q = patch_idx_q[valid_mask]
    
    patch_corrs_map = torch.zeros(N, N)
    
    for i in range(valid_idx_a.shape[0]):
        patch_corrs_map[valid_idx_a[i], valid_idx_q[i]] += 1
    # patch_corrs_map.index_add_(0, valid_idx_a, torch.nn.functional.one_hot(valid_idx_q, num_classes=N).float())

    row_sums = patch_corrs_map.sum(dim=1, keepdim=True)
    row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
    patch_corrs_map = patch_corrs_map / row_sums

    return patch_corrs_map

class CollateWrapperPCM(CollateWrapper):
    def __init__(self, corr_n : int):
        super().__init__(corr_n)
        
    def __call__(self, data):
        final_dict = super().__call__(data)
        temp_lst_a_dict = {}
        temp_lst_q_dict = {}
        key_lst = ['orig_mask', 'orig_boxes', 'pred_boxes', 'pred_mask', 'cropped_pred_mask', 'patch_corrs_map']
        for key in key_lst:
            temp_lst_a_dict[key] = []
            temp_lst_q_dict[key] = []
        # pred_box_a_lst, pred_mask_a_lst, pred_box_q_lst, pred_mask_q_lst = [], [], [], []
        for item_a, item_q, *other_data in data:
            for key in key_lst:
                value_a = item_a[key]
                value_a = value_a if isinstance(value_a, torch.Tensor) else torch.tensor(value_a)
                temp_lst_a_dict[key].append(value_a)

                value_q = item_q[key]
                value_q = value_q if isinstance(value_q, torch.Tensor) else torch.tensor(value_q)
                temp_lst_q_dict[key].append(value_q)

        for key in key_lst:
            final_dict['anchor'][key] = torch.stack(temp_lst_a_dict[key], dim=0)
            final_dict['query'][key] = torch.stack(temp_lst_q_dict[key], dim=0)

        return final_dict

class Shapenet6DDatasetPCM(Shapenet6DDataset):
    def __init__(self, args, eval=False):
        super().__init__(args, eval)
        self.collate = CollateWrapperPCM(self.max_corrs)
        self.patch_corrs_grid_size = args.dataset.patch_corrs_grid_size
        if eval:
            self.mask_predictor = OvMaskPredictor('shapenet6d', load_det_model=False, load_sam_model=False)
        else:
            self.mask_predictor = None

    def __getitem__(self, index, i=0):
        img_a, img_q, cat_id = self.instances[index]
        instance_id = f'{img_a}_{img_q}_{cat_id}'
        #print(scene_id_a, img_id_a, ', ', scene_id_q, img_id_q, ', ', cat_id)
        orig_corrs = self.corrs[index]
        pose = self.poses[index]
        
        path = join(self.root, self.name)
        item_a = shapenet.get_item_data(path, self.annots, self.metadata, img_a, cat_id)
        item_q = shapenet.get_item_data(path, self.annots, self.metadata, img_q, cat_id)
                
        item_a = common.preprocess_item(item_a)
        item_q = common.preprocess_item(item_q)

        # prompt is the same by construction
        prompt = self.get_item_prompt(item_a)
        orig_corrs = torch.tensor(orig_corrs)
        # viz.corr_set(item_a['rgb'], item_q['rgb'], orig_corrs.numpy(), orig_corrs.numpy(), 'ppp1.png')

        item_a = crop_item_img_via_pred_box(item_a, self.mask_predictor, eval=self.eval)
        item_q = crop_item_img_via_pred_box(item_q, self.mask_predictor, eval=self.eval)
        new_corrs = get_new_corrs_after_crop(item_a, item_q, orig_corrs)

        item_a, item_q, res_corrs = self.augs_fn((item_a, item_q, new_corrs))
        # viz.corr_set(item_a['rgb'], item_q['rgb'], res_corrs.numpy(), res_corrs.numpy(), 'ppp2.png')         

        sampled_corrs, valid_corrs = sample_correspondences(res_corrs, instance_id, self.debug_valid, self.max_corrs)
        # viz.corr_set(item_a['rgb'], item_q['rgb'], sampled_corrs.numpy(), sampled_corrs.numpy(), 'ppp3.png')
        # viz.pred_mask(item_a['rgb'].numpy().transpose(1, 2, 0), item_q['rgb'].numpy().transpose(1, 2, 0), item_a['mask'].numpy(), item_q['mask'].numpy(), item_a['cropped_pred_mask'].numpy(), item_q['cropped_pred_mask'].numpy(), torch.zeros_like(item_a['cropped_pred_mask']).numpy(), torch.zeros_like(item_q['cropped_pred_mask']).numpy(), 'ppp4.png')
        
        valid_a = common.check_validity(item_a)
        valid_q = common.check_validity(item_q)
        valid = valid_a and valid_q and valid_corrs

        patch_corrs_map =  compute_patch_corrs_map(item_a['rgb'].shape[1:], sampled_corrs, self.patch_corrs_grid_size)
        item_a['patch_corrs_map'] = patch_corrs_map
        item_q['patch_corrs_map'] = patch_corrs_map

        return item_a, item_q, prompt, sampled_corrs, orig_corrs, pose, cat_id, instance_id, valid


class NOCSDatasetPCM(NOCSDataset):
    def __init__(self, args, eval=False):
        super().__init__(args, eval)
        self.collate = CollateWrapperPCM(self.max_corrs)
        self.patch_corrs_grid_size = args.dataset.patch_corrs_grid_size
        if eval:
            self.mask_predictor = OvMaskPredictor('nocs', load_det_model=False, load_sam_model=False)
        else:
            self.mask_predictor = None

    def __getitem__(self, index):
        split, scene_a, img_a, scene_q, img_q, cat_id, obj_id = self.instances[index]
        instance_id = f'{scene_a}_{img_a}_{scene_q}_{img_q}_{obj_id}'
        orig_corrs = self.corrs[index]
        pose = self.poses[index]
        path = join(self.root, self.name)
        mask = get_mask_type(self.mask_type, self.eval)
        
        item_a = nocs.get_item_data(path, scene_a, img_a, self.abs_poses, self.obj_names, obj_id, mask)
        item_q = nocs.get_item_data(path, scene_q, img_q, self.abs_poses, self.obj_names, obj_id, mask)

        item_a['camera'] = self.K 
        item_q['camera'] = self.K

        item_a = common.preprocess_item(item_a)
        item_q = common.preprocess_item(item_q)
        # prompt is the same by construction
        prompt = self.get_item_prompt(item_a)
        orig_corrs = torch.tensor(orig_corrs)
        # viz.corr_set(item_a['rgb'], item_q['rgb'], orig_corrs.numpy(), orig_corrs.numpy(), 'ppp1.png')

        item_a = crop_item_img_via_pred_box(item_a, self.mask_predictor, eval=self.eval)
        item_q = crop_item_img_via_pred_box(item_q, self.mask_predictor, eval=self.eval)
        new_corrs = get_new_corrs_after_crop(item_a, item_q, orig_corrs)

        item_a, item_q, res_corrs = self.augs_fn((item_a, item_q, new_corrs))
        # viz.corr_set(item_a['rgb'], item_q['rgb'], res_corrs.numpy(), res_corrs.numpy(), 'ppp2.png') 

        sampled_corrs, valid_corrs = sample_correspondences(res_corrs, instance_id, self.debug_valid, self.max_corrs)
        # viz.corr_set(item_a['rgb'], item_q['rgb'], sampled_corrs.numpy(), sampled_corrs.numpy(), 'ppp3.png')
        # viz.pred_mask(item_a['rgb'].numpy().transpose(1, 2, 0), item_q['rgb'].numpy().transpose(1, 2, 0), item_a['mask'].numpy(), item_q['mask'].numpy(), item_a['cropped_pred_mask'].numpy(), item_q['cropped_pred_mask'].numpy(), torch.zeros_like(item_a['cropped_pred_mask']).numpy(), torch.zeros_like(item_q['cropped_pred_mask']).numpy(), 'ppp4.png')

        # unvalid objects are skipped at training time and counted as automatic failure at test times 
        # this should only happen when using a predictede segm mask, e.g. ovseg  
        valid_a = common.check_validity(item_a)
        valid_q = common.check_validity(item_q)
        valid = valid_a and valid_q and valid_corrs

        patch_corrs_map =  compute_patch_corrs_map(item_a['rgb'].shape[1:], sampled_corrs, self.patch_corrs_grid_size)
        item_a['patch_corrs_map'] = patch_corrs_map
        item_q['patch_corrs_map'] = patch_corrs_map

        return item_a, item_q, prompt, sampled_corrs, orig_corrs, pose, obj_id, instance_id, valid

class TOYLDatasetPCM(TOYLDataset):
    def __init__(self, args, eval=False):
        super().__init__(args, eval)
        self.collate = CollateWrapperPCM(self.max_corrs)
        self.patch_corrs_grid_size = args.dataset.patch_corrs_grid_size
        if eval:
            self.mask_predictor = OvMaskPredictor('toyl', load_det_model=False, load_sam_model=False)
        else:
            self.mask_predictor = None

    def __getitem__(self, index):
        split, scene_a, img_a, scene_q, img_q, cls_id = self.instances[index]
        instance_id = f'{scene_a}_{img_a}_{scene_q}_{img_q}_{cls_id}'
        orig_corrs = self.corrs[index]
        pose = self.poses[index]

        mask_type = get_mask_type(self.mask_type, self.eval)
        item_a = toyl.get_item_data(self.local_root, scene_a, img_a, self.part_data, self.obj_names, cls_id, mask_type)
        item_q = toyl.get_item_data(self.local_root, scene_q, img_q, self.part_data, self.obj_names, cls_id, mask_type)
        
        item_a['camera'] = self.K 
        item_q['camera'] = self.K

        item_a = common.preprocess_item(item_a)
        item_q = common.preprocess_item(item_q)
        
        orig_corrs = torch.tensor(orig_corrs)
        # prompt is the same by construction
        prompt = self.get_item_prompt(item_a)

        # viz.corr_set(item_a['rgb'], item_q['rgb'], orig_corrs.numpy(), orig_corrs.numpy(), 'ppp1.png')

        item_a = crop_item_img_via_pred_box(item_a, self.mask_predictor, eval=self.eval)
        item_q = crop_item_img_via_pred_box(item_q, self.mask_predictor, eval=self.eval)
        new_corrs = get_new_corrs_after_crop(item_a, item_q, orig_corrs)

        item_a, item_q, res_corrs = self.augs_fn((item_a, item_q, new_corrs))
        # viz.corr_set(item_a['rgb'], item_q['rgb'], res_corrs.numpy(), res_corrs.numpy(), 'ppp2.png') 

        sampled_corrs, valid_corrs = sample_correspondences(res_corrs, instance_id, self.debug_valid, self.max_corrs)
        # viz.corr_set(item_a['rgb'], item_q['rgb'], sampled_corrs.numpy(), sampled_corrs.numpy(), 'ppp3.png')
        
        # unvalid objects are skipped at training time and counted as automatic failure at test times   
        valid_a = common.check_validity(item_a)
        valid_q = common.check_validity(item_q)
        valid = valid_a and valid_q and valid_corrs

        patch_corrs_map =  compute_patch_corrs_map(item_a['rgb'].shape[1:], sampled_corrs, self.patch_corrs_grid_size)
        item_a['patch_corrs_map'] = patch_corrs_map
        item_q['patch_corrs_map'] = patch_corrs_map

        return item_a, item_q, prompt, sampled_corrs, orig_corrs, pose, cls_id, instance_id, valid
