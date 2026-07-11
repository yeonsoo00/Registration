"""Grouped affine inference: one predicted matrix is applied to every stain in a group.



python inference.py \
  --checkpoint /home/yec23006/projects/research/Registration/Grouped/ckpt/group_overlay_depth6_512/best_model.pt \
  --registered_root /home/yec23006/projects/research/Registration/Data/Testdata/Registered \
  --unregistered_root /home/yec23006/projects/research/Registration/Data/Testdata/Unregistered \
  --output_dir /home/yec23006/projects/research/Registration/Grouped/Results/group_overlay_depth6_512 \
  --batch_size 4 \
  --n_workers 4 \
  --device cuda \
  --gpu_ids 0,1
  
  """
from __future__ import annotations

import argparse, csv, os
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from dataset import CartilageDataset
from models import GroupAffineRegistrationModel
from utils import affine_parameters_to_matrix, compute_mineral_mask, compute_preprocess_geometry, load_image, normalized_affine_to_pixel_matrix




def safe_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise RuntimeError("Cannot collate an empty batch")
    result: Dict[str, Any] = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        first = values[0]
        if torch.is_tensor(first):
            shapes = [tuple(value.shape) for value in values]
            if any(shape != shapes[0] for shape in shapes[1:]):
                raise RuntimeError(f"Variable tensor shape for key '{key}': {shapes}")
            result[key] = torch.stack(
                [value.detach().contiguous().clone() for value in values], dim=0
            )
        else:
            result[key] = values
    return result

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True); p.add_argument('--registered_root',required=True); p.add_argument('--unregistered_root',required=True); p.add_argument('--output_dir',required=True)
    p.add_argument('--batch_size',type=int,default=4); p.add_argument('--n_workers',type=int,default=2); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--gpu_ids',default='0,1'); p.add_argument('--no_multi_gpu',action='store_true')
    return p.parse_args()


def save_rgb(path,img):
    os.makedirs(os.path.dirname(path),exist_ok=True); Image.fromarray(img.astype(np.uint8),'RGB').save(path)


def make_overlay(images: List[np.ndarray]) -> np.ndarray:
    if not images: raise ValueError('No images for overlay')
    arr=np.stack([x.astype(np.float32) for x in images],axis=0)
    return np.clip(arr.max(axis=0),0,255).astype(np.uint8)


def main(a):
    device=torch.device(a.device); ck=torch.load(a.checkpoint,map_location=device); pre=ck['preprocess_config']; cfg=ck['model_config']
    ds=CartilageDataset(a.registered_root,a.unregistered_root,size=(pre['height'],pre['width']),image_mode=pre['image_mode'],sfo_mode=pre['sfo_mode'],crop_mode=pre['crop_mode'],crop_margin=pre['crop_margin'],synthetic_prob=0.0,deterministic_synthetic=True,group_input_mode=pre['group_input_mode'],include_group1=pre.get('include_group1',True))
    dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.n_workers,pin_memory=device.type=='cuda')
    model=GroupAffineRegistrationModel(**cfg).to(device); state=OrderedDict((k[7:] if k.startswith('module.') else k,v) for k,v in ck['model_state_dict'].items()); model.load_state_dict(state)
    if device.type=='cuda' and not a.no_multi_gpu:
        ids=[int(x) for x in a.gpu_ids.split(',') if x.strip()]
        if len(ids)>1: model=torch.nn.DataParallel(model,device_ids=ids)
    model.eval(); aligned_root=os.path.join(a.output_dir,'aligned_original_rgb'); overlay_root=os.path.join(a.output_dir,'group_overlays'); os.makedirs(aligned_root,exist_ok=True); os.makedirs(overlay_root,exist_ok=True)
    rows=[]; offset=0; grouped_outputs=defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for batch in dl:
            gi=batch['group_input'].to(device); fixed=batch['fixed_mineral'].to(device); group=batch['group'].to(device); params=model(gi,fixed,group); mats=affine_parameters_to_matrix(params)
            bs=gi.shape[0]
            for j in range(bs):
                ds_idx=offset+j; sample_name,group_id,_=ds.items[ds_idx]; sample=ds.samples[sample_name]; reg_stains=sample['reg_stains']; unreg_stains=sample['unreg_stains']; mineral_path=str(sample['mineral'])
                fixed_gray=load_image(mineral_path,grayscale=True); fixed_h,fixed_w=fixed_gray.shape; mask=compute_mineral_mask(fixed_gray); geom=compute_preprocess_geometry(mask,(pre['height'],pre['width']),crop_mode=pre['crop_mode'],crop_margin=pre['crop_margin']); pre_m=geom.original_to_model_matrix(); pre_inv=np.linalg.inv(pre_m); model_px=normalized_affine_to_pixel_matrix(mats[j],pre['height'],pre['width'])
                stain_indices=batch['stain_indices'][j].tolist(); valid=batch['valid_group'][j].tolist()
                for stain_idx,is_valid in zip(stain_indices,valid):
                    if not is_valid: continue
                    moving_path=unreg_stains.get(int(stain_idx))
                    if moving_path is None: continue
                    moving_rgb=load_image(str(moving_path),grayscale=False); mh,mw=moving_rgb.shape[:2]
                    fixed_canvas_to_moving=np.array([[mw/fixed_w,0,0],[0,mh/fixed_h,0],[0,0,1]],dtype=np.float64)
                    dst_to_src=fixed_canvas_to_moving @ pre_inv @ model_px @ pre_m
                    aligned=cv2.warpAffine(np.clip(moving_rgb*255,0,255).astype(np.uint8),dst_to_src[:2],(fixed_w,fixed_h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
                    out=os.path.join(aligned_root,sample_name,f'group{int(group_id)}_stain{int(stain_idx)}_aligned.png'); save_rgb(out,aligned); grouped_outputs[sample_name][int(group_id)].append(aligned)
                    p=params[j].cpu().numpy(); rows.append({'sample':sample_name,'group':int(group_id),'stain_idx':int(stain_idx),'tx_normalized':float(p[0]),'ty_normalized':float(p[1]),'rotation_degrees':float(np.rad2deg(p[2])),'scale_x':float(p[3]),'scale_y':float(p[4]),'output':out})
            offset+=bs
    for sample,groups in grouped_outputs.items():
        for gid,imgs in groups.items(): save_rgb(os.path.join(overlay_root,sample,f'group{gid}_overlay.png'),make_overlay(imgs))
    with open(os.path.join(a.output_dir,'predicted_group_affine_parameters.csv'),'w',newline='') as f:
        fields=list(rows[0]) if rows else ['sample']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f'Saved {len(rows)} aligned stain images and group overlays to {a.output_dir}')

if __name__=='__main__': main(parse_args())
