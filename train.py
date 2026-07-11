"""Train one affine transform per stain acquisition group.

python train.py \
  --registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered \
  --unregistered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Unregistered \
  --output_dir /home/yec23006/projects/research/Registration/Grouped/ckpt/group_stack \
  --group_input_mode stack \
  --use_group_embedding \
  --height 512 \
  --width 512 \
  --image_mode rgb \
  --sfo_mode rgb \
  --crop_mode full \
  --fusion_mode intermediate \
  --depth 4 \
  --base_channels 32 \
  --latent_dim 256 \
  --group_embedding_dim 32 \
  --spatial_pool_size 4 \
  --norm_type group \
  --synthetic_prob 1.0 \
  --val_synthetic_prob 1.0 \
  --tx_range -100 100 \
  --ty_range -1000 1000 \
  --rot_range -20 20 \
  --scale_range 0.8 1.2 \
  --model_scale_range 0.8 1.2 \
  --translation_limit 0.5 \
  --max_rotation_deg 20 \
  --param_weight 5.0 \
  --ncc_weight 1.0 \
  --edge_weight 0.5 \
  --ssd_weight 0.0 \
  --reg_weight 0.0 \
  --epochs 200 \
  --batch_size 4 \
  --lr 0.0001 \
  --weight_decay 0.00001 \
  --amp \
  --gpu_ids 0,1 \
  --wandb_project registration \
  --wandb_run_name group_stack_affine

python train.py \
  --registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered \
  --unregistered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Unregistered \
  --output_dir /home/yec23006/projects/research/Registration/Grouped/ckpt/group_overlay \
  --group_input_mode overlay \
  --use_group_embedding \
  --height 512 \
  --width 512 \
  --image_mode rgb \
  --sfo_mode rgb \
  --crop_mode full \
  --fusion_mode intermediate \
  --depth 4 \
  --base_channels 32 \
  --latent_dim 256 \
  --group_embedding_dim 32 \
  --spatial_pool_size 4 \
  --norm_type group \
  --synthetic_prob 1.0 \
  --val_synthetic_prob 1.0 \
  --param_weight 5.0 \
  --ncc_weight 1.0 \
  --edge_weight 0.5 \
  --ssd_weight 0.0 \
  --reg_weight 0.0 \
  --epochs 200 \
  --batch_size 4 \
  --lr 0.0001 \
  --amp \
  --gpu_ids 0,1 \
  --wandb_project registration \
  --wandb_run_name group_overlay_affine

"""

from __future__ import annotations

import argparse, os, random
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import CartilageDataset
from losses import gradient_ncc_loss, ncc_loss, param_loss, regularisation_loss, ssd_loss
from models import GroupAffineRegistrationModel
from utils import affine_parameters_to_matrix, apply_affine_transform


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--registered_root',required=True); p.add_argument('--unregistered_root',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--epochs',type=int,default=200); p.add_argument('--batch_size',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--weight_decay',type=float,default=1e-5)
    p.add_argument('--n_workers',type=int,default=4); p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--gpu_ids',default='0,1'); p.add_argument('--no_multi_gpu',action='store_true'); p.add_argument('--amp',action='store_true')
    p.add_argument('--height',type=int,default=512); p.add_argument('--width',type=int,default=512)
    p.add_argument('--image_mode',choices=['rgb','gray'],default='rgb'); p.add_argument('--sfo_mode',choices=['rgb','hsv','gray'],default='rgb')
    p.add_argument('--crop_mode',choices=['full','mineral_bbox'],default='full'); p.add_argument('--crop_margin',type=int,default=32)
    p.add_argument('--group_input_mode',choices=['single','stack','overlay'],default='stack')
    p.add_argument('--use_group_embedding',action='store_true'); p.add_argument('--include_group1',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--depth',type=int,default=4); p.add_argument('--base_channels',type=int,default=32); p.add_argument('--latent_dim',type=int,default=256)
    p.add_argument('--group_embedding_dim',type=int,default=32); p.add_argument('--spatial_pool_size',type=int,default=4)
    p.add_argument('--norm_type',choices=['group','batch','instance'],default='group'); p.add_argument('--fusion_mode',choices=['concat','intermediate'],default='intermediate')
    p.add_argument('--disable_coordconv',action='store_true'); p.add_argument('--no_force_group1_identity',action='store_true')
    p.add_argument('--model_scale_range',type=float,nargs=2,default=[0.8,1.2]); p.add_argument('--translation_limit',type=float,default=0.5); p.add_argument('--max_rotation_deg',type=float,default=20)
    p.add_argument('--synthetic_prob',type=float,default=1.0); p.add_argument('--val_synthetic_prob',type=float,default=1.0)
    p.add_argument('--tx_range',type=float,nargs=2,default=[-100,100]); p.add_argument('--ty_range',type=float,nargs=2,default=[-500,500]); p.add_argument('--rot_range',type=float,nargs=2,default=[-20,20]); p.add_argument('--scale_range',type=float,nargs=2,default=[0.8,1.2])
    p.add_argument('--param_weight',type=float,default=5.0); p.add_argument('--ncc_weight',type=float,default=1.0); p.add_argument('--edge_weight',type=float,default=0.5); p.add_argument('--ssd_weight',type=float,default=0.0); p.add_argument('--reg_weight',type=float,default=0.0)
    p.add_argument('--val_split',type=float,default=0.15); p.add_argument('--split_seed',type=int,default=2026)
    p.add_argument('--resume_checkpoint'); p.add_argument('--wandb_project'); p.add_argument('--wandb_run_name')
    return p.parse_args()




def safe_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack fixed-shape grouped samples without shared-memory resize.

    PyTorch's default worker-side collate preallocates non-resizable shared
    storage. If any item has a different shape, the resulting error is the
    opaque ``Trying to resize storage that is not resizable``. This collate
    function checks every tensor key explicitly and then calls ``torch.stack``
    without the shared ``out=`` buffer, producing an informative shape error.
    """
    if not batch:
        raise RuntimeError("Cannot collate an empty batch")

    result: Dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [sample[key] for sample in batch]
        first = values[0]
        if torch.is_tensor(first):
            shapes = [tuple(value.shape) for value in values]
            dtypes = [value.dtype for value in values]
            if any(shape != shapes[0] for shape in shapes[1:]):
                raise RuntimeError(
                    f"Variable tensor shape for key '{key}': {shapes}. "
                    "All grouped samples must use fixed padded dimensions."
                )
            if any(dtype != dtypes[0] for dtype in dtypes[1:]):
                raise RuntimeError(
                    f"Variable tensor dtype for key '{key}': {dtypes}"
                )
            result[key] = torch.stack(
                [value.detach().contiguous().clone() for value in values], dim=0
            )
        else:
            result[key] = values
    return result

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def build_dataset(a,validation=False):
    return CartilageDataset(a.registered_root,a.unregistered_root,size=(a.height,a.width),image_mode=a.image_mode,sfo_mode=a.sfo_mode,crop_mode=a.crop_mode,crop_margin=a.crop_margin,synthetic_prob=a.val_synthetic_prob if validation else a.synthetic_prob,tx_range=tuple(a.tx_range),ty_range=tuple(a.ty_range),rot_range=tuple(a.rot_range),scale_range=tuple(a.scale_range),deterministic_synthetic=validation,synthetic_seed=9090,group_input_mode=a.group_input_mode,include_group1=a.include_group1)


def split(ds,frac,seed):
    samples=sorted({x[0] for x in ds.items}); rng=np.random.default_rng(seed); rng.shuffle(samples)
    n=max(1,int(round(len(samples)*frac))); vs=set(samples[:n])
    tr=[i for i,x in enumerate(ds.items) if x[0] not in vs]; va=[i for i,x in enumerate(ds.items) if x[0] in vs]
    return tr,va


def warp_group(moving,params):
    # moving: B,K,C,H,W ; same matrix repeated for all K members
    b,k,c,h,w=moving.shape
    mats=affine_parameters_to_matrix(params)
    mats=mats[:,None].expand(b,k,2,3).reshape(b*k,2,3)
    out=apply_affine_transform(moving.reshape(b*k,c,h,w),mats)
    return out.reshape(b,k,c,h,w)


def grouped_loss(a,params,warped,batch):
    valid=batch['valid_group'].bool(); target=batch['target_group']; b,k,c,h,w=warped.shape
    flat_valid=valid.reshape(-1)
    if not flat_valid.any(): raise RuntimeError('Batch contains no valid group members')
    wf=warped.reshape(b*k,c,h,w)[flat_valid]; tf=target.reshape(b*k,c,h,w)[flat_valid]
    losses={}
    if a.ncc_weight: losses['ncc']=ncc_loss(wf,tf)
    if a.edge_weight: losses['edge']=gradient_ncc_loss(wf,tf)
    if a.ssd_weight: losses['ssd']=ssd_loss(wf,tf)
    hp=batch['has_params'].reshape(-1).bool()
    if a.param_weight and hp.any(): losses['param']=param_loss(params[hp],batch['params_true'][hp])
    if a.reg_weight: losses['reg']=regularisation_loss(params)
    weights={'ncc':a.ncc_weight,'edge':a.edge_weight,'ssd':a.ssd_weight,'param':a.param_weight,'reg':a.reg_weight}
    total=sum(weights[n]*v for n,v in losses.items()); losses['total']=total
    return total,losses


def evaluate(a,model,loader,device):
    model.eval(); sums={}; n=0
    with torch.no_grad():
        for batch in loader:
            batch={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
            p=model(batch['group_input'],batch['fixed_mineral'],batch['group']); w=warp_group(batch['moving_group'],p)
            _,ls=grouped_loss(a,p,w,batch); bs=p.shape[0]
            for k,v in ls.items(): sums['val_'+k]=sums.get('val_'+k,0)+float(v)*bs
            n+=bs
    return {k:v/max(n,1) for k,v in sums.items()}


def main(a):
    set_seed(a.seed); os.makedirs(a.output_dir,exist_ok=True); device=torch.device(a.device)
    trb=build_dataset(a,False); vab=build_dataset(a,True); tri,vai=split(trb,a.val_split,a.split_seed)
    tr=DataLoader(Subset(trb,tri),batch_size=a.batch_size,shuffle=True,num_workers=a.n_workers,pin_memory=device.type=='cuda',collate_fn=safe_collate,persistent_workers=a.n_workers>0)
    va=DataLoader(Subset(vab,vai),batch_size=a.batch_size,shuffle=False,num_workers=a.n_workers,pin_memory=device.type=='cuda',collate_fn=safe_collate,persistent_workers=a.n_workers>0)
    cfg=dict(group_input_channels=trb.group_input_channels,fixed_channels=trb.channels,latent_dim=a.latent_dim,group_embedding_dim=a.group_embedding_dim,use_group_embedding=a.use_group_embedding,scale_range=tuple(a.model_scale_range),translation_limit=a.translation_limit,max_rotation_degrees=a.max_rotation_deg,depth=a.depth,base_channels=a.base_channels,spatial_pool_size=a.spatial_pool_size,norm_type=a.norm_type,use_coordconv=not a.disable_coordconv,fusion_mode=a.fusion_mode,force_group1_identity=not a.no_force_group1_identity)
    model=GroupAffineRegistrationModel(**cfg).to(device)
    if device.type=='cuda' and not a.no_multi_gpu:
        ids=[int(x) for x in a.gpu_ids.split(',') if x.strip()]
        if len(ids)>1: model=torch.nn.DataParallel(model,device_ids=ids)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    if a.resume_checkpoint:
        ck=torch.load(a.resume_checkpoint,map_location=device); state=ck.get('model_state_dict',ck); state=OrderedDict((k[7:] if k.startswith('module.') else k,v) for k,v in state.items()); (model.module if isinstance(model,torch.nn.DataParallel) else model).load_state_dict(state)
    scaler=torch.cuda.amp.GradScaler(enabled=a.amp and device.type=='cuda'); best=float('inf')
    use_wandb=bool(a.wandb_project)
    if use_wandb:
        import wandb; wandb.init(project=a.wandb_project,name=a.wandb_run_name,config=vars(a))
    for epoch in range(1,a.epochs+1):
        model.train(); sums={}; n=0
        for batch in tqdm(tr,desc=f'Epoch {epoch}/{a.epochs}',leave=False):
            batch={k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}; opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=a.amp and device.type=='cuda'):
                p=model(batch['group_input'],batch['fixed_mineral'],batch['group']); w=warp_group(batch['moving_group'],p); total,ls=grouped_loss(a,p,w,batch)
            scaler.scale(total).backward(); scaler.step(opt); scaler.update(); bs=p.shape[0]; n+=bs
            for k,v in ls.items(): sums['train_'+k]=sums.get('train_'+k,0)+float(v.detach())*bs
        met={k:v/max(n,1) for k,v in sums.items()}; met.update(evaluate(a,model,va,device)); met['epoch']=epoch
        print(' '.join(f'{k}={v:.6f}' for k,v in met.items() if k in {'train_total','train_param','val_total','val_param','val_ncc'}))
        if use_wandb:
            import wandb; wandb.log(met)
        tm=model.module if isinstance(model,torch.nn.DataParallel) else model
        payload={'model_state_dict':tm.state_dict(),'optimizer_state_dict':opt.state_dict(),'epoch':epoch,'metrics':met,'model_config':cfg,'preprocess_config':{'height':a.height,'width':a.width,'image_mode':a.image_mode,'sfo_mode':a.sfo_mode,'crop_mode':a.crop_mode,'crop_margin':a.crop_margin,'group_input_mode':a.group_input_mode,'include_group1':a.include_group1}}
        torch.save(payload,os.path.join(a.output_dir,'last_model.pt'))
        metric=met.get('val_total',met['train_total'])
        if metric<best: best=metric; torch.save(payload,os.path.join(a.output_dir,'best_model.pt'))
    if use_wandb:
        import wandb; wandb.finish()

if __name__=='__main__': main(parse_args())
