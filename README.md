# Teacher/student correlation-volume registration

This is an *TeachReg* on the Correlation-Volume network with teacher-student framework. 

The student model is the main model. It predicts one affine transform for
an acquisition group from registered Mineral, the unregistered moving stains,
and the group ID:

```python
student_params = model(
    fixed_mineral=fixed_mineral,
    moving_group=moving_group,
    group=group_id,
)
```

It never receives a registered target stain. Normal inference therefore needs
only Mineral reference image and unregistered stains.

During training, an independent teacher uses both the Mineral reference image and the easier registered and unregistered same-stain target:

```python
teacher_params = model.forward_teacher(
    fixed_mineral=fixed_mineral,
    target_group=target_group,
    moving_group=moving_group,
    group=group_id,
)
```

`target_group` contains the registered images of `moving_group`. 
It is allowed in this training-only teacher call and in the loss, but it is never passed to the student's `forward()` method or required
by normal inference.

## Registration and group contract

| Group | Signals |
| --- | --- |
| G1 | Mineral (1), AC (2), Calcein (3) |
| G2 | TRAP (4) |
| G3 | DAPI (5), AP (6), EDU (7) |
| G4 | CFO (8) |
| G5 | SFO (9) |

Every valid stain in a group is warped with the same predicted affine matrix. Mineral
is never warped. AC and Calcein were acquired with Mineral, so they don't warp. 
The dataset retains the deployable eight-field contract:

## Architecture

The two paths have independent learnable correlation networks and begin with
the same weights. They use the same configured frontend, but different
fixed-side sources:

```text
[Deployable student]

 Mineral Reference                Moving Group
        │                               │
 Optional Fixed Adapter        Group-Specific Adapter
        │                               │
        └──── Shared-Weight FPN ────────┘
                        │
             Local Correlation Volumes
                        │
              Group-Specific Affine Head
                        │
              Student Affine Transform ^A
                        │
             Warp(Moving Group, ^A)
                        │
        Image Loss vs Registered Target Group


[Training-only teacher]


 Registered Mineral       Registered Target Group              Moving Group
         │                           │                                │
         └── Evidence-Aware Fusion ──┘                          Group Frontend
                       │                                              │
               Teacher Fixed Side             +               Teacher Moving Side
                                              │
                                Independent Shared-Weight FPN
                                              │
                                  Local Correlation Volumes
                                              │
                                    Group-Specific Affine Head
                                              │
                                  Teacher Affine Transform A_T
```


Each output is `{"params": affine_params, "levels": [...]}`. Every selected
level contains:

| Field | Meaning |
| --- | --- |
| `volume` | raw pre-softmax cosine cost volume, `B x Kd x H x W` |
| `probability` | validity-masked correspondence probability over `Kd` local displacements |
| `expected_displacement` | ungated probability expectation `(dx, dy)` in feature pixels |
| `confidence` | maximum candidate probability |
| `certainty` | one minus normalized correspondence entropy |
| `displacements` | candidate table in `(dx, dy)` order |
| `fixed_valid`, `moving_valid` | evidence/FOV validity at this pyramid level |
| `feature_size` | `[H, W]` for the level |


Synthetic `params_true` uses the existing PyTorch sampling convention. Let
`A_true = affine_parameters_to_matrix(params_true)`. Because `affine_grid`
maps output coordinates to input coordinates, `A_true` maps a fixed feature
coordinate to the corresponding moving feature coordinate. For each level:

```text
q_fixed  = identity affine_grid coordinate
q_moving = A_true(q_fixed)
d_true   = (q_moving - q_fixed) * ((W - 1) / 2, (H - 1) / 2)
```

Thus `d_true` is a fixed-to-moving displacement in feature-map pixels, exactly
matching `C(x,d) = cosine(F_fixed(x), F_moving(x+d))`. The target builder uses
`A_true` directly, not its inverse, and uses `align_corners=True` just like the
registration utilities.

The geometric confidence mask requires a synthetic row (`has_params=True`), a
valid fixed location, an in-image `q_moving`, and moving validity sampled at
that coordinate. Displacement and distribution supervision further require
both components of `d_true` to lie inside the square local search radius. The
Gaussian distribution target is

```text
p_target(d_i) proportional to exp(-||d_i - d_true||^2 / (2 sigma^2))
```

and is normalized only over `candidate_valid` displacements. Empty masks and
levels are skipped with finite graph-connected zeros rather than NaNs. Real
rows keep their NaN `params_true` sentinel and never enter affine conversion or
correspondence supervision.

The three direct terms are:

- Smooth L1 between predicted and true expected displacement;
- soft cross-entropy between predicted probability and the Gaussian target;
- binary cross-entropy on confidence. A geometrically valid in-radius match has
  confidence target 1; a geometrically valid match outside the search radius
  has target 0. Out-of-image correspondences are excluded rather than labeled
  negative.

For branch `b` (`student` or `teacher`), weighting is:

```text
L_corr_b = b_corr_weight * (
    corr_displacement_weight * L_displacement
  + corr_distribution_weight * L_distribution
  + corr_confidence_weight   * L_confidence
)
```

`--corr_target_sigma` is measured in feature pixels and defaults to `1.0`.
`--student_corr_weight` and `--teacher_corr_weight` default to `1.0` and allow
one branch to be ablated without changing the component weights. All three
component weights default to zero. 
Synthetic samples supervise student and teacher independently. During teacher-only warmup only
the teacher term updates, while the normal phase can apply both branch terms.


## Training objective

Outside teacher-only warmup, the student supplies the image-registration
prediction:

```python
student_params = model(
    fixed_mineral=fixed_mineral,
    moving_group=moving_group,
    group=group_id,
)
# Real Stage 2 path; synthetic Stage 1 uses the composed one-pass path below.
corrected_group = warp_group(real_moving_group, student_params)
image_loss = image_objective(corrected_group, target_group, valid_group)
```

Registered targets are used only after prediction. Valid-stain and
signal-support masks remove padded slots and newly exposed affine borders. The
image objective can combine multiscale local NCC, gradient NCC, Charbonnier
distance, multiscale gradient loss, and soft foreground Dice.


`L_teacher_image` uses the same NCC, edge, Charbonnier, gradient, overlap, and
valid-FOV machinery as the student image objective. Empty labeled or unlabeled
subsets are skipped, so `synthetic_prob` may be anywhere in `[0, 1]` and
`param_weight` may be zero.

For synthetic Stage 1, image supervision and validation images avoid applying a second interpolating warp to an already transformed
and clipped tensor. Let `target_group` be the registered source before the
synthetic augmentation and let:

```text
A_syn = inverse(affine_parameters_to_matrix(params_true))
synthetic_moving = warp(target_group, A_syn)
corrected_sequential = warp(synthetic_moving, A_pred)
A_final = A_syn @ A_pred
corrected_full = warp(target_group, A_final)
```


Real Stage 2 has no known synthetic affine or pre-augmentation registered
source for the moving observation, so its behavior is unchanged:

```python
corrected_group = warp_group(real_moving_group, student_params)
```

### Stage 1 : Knowledge distillation training

This complete run activates only the two additions in this experiment: the residual FPN and direct supervision of the existing local cost volumes.
It uses the separated affine head.

```
python train.py \
--registered_root <PATH/TO/DATA> \
--unregistered_root <PATH/TO/DATA> \
--output_dir <PATH/TO/DIR> \
--best_checkpoint_name best_model.pt --last_checkpoint_name last_model.pt \
--use_teacher_branch --frontend_mode raw --group_input_mode overlay \
--affine_head_mode separated \
--teacher_distill_weight 1.0 --detach_teacher --teacher_warmup_epochs 50 \
--no-include_group1 --force_group1_identity --separate_group_heads --separate_group_adapters --use_group_embedding \
--height 1024 --width 1024 --image_mode rgb --sfo_mode hsv \
--crop_mode full --crop_margin 32 \
--encoder_arch residual --encoder_channels 64 96 128 192 256 \
--encoder_blocks_per_stage 2 2 2 2 2 --encoder_base_channels 24 \
--encoder_depth 5 --feature_width 64 --correlation_feature_width 128 \
--cost_hidden_channels 48 --cost_volume_radii 8 6 4 --cost_pool_size 4 \
--correlation_temperature 0.07 --latent_dim 512 \
--group_embedding_dim 32 --norm_type group --model_scale_range 0.8 1.2 \
--translation_limit 0.5 --max_rotation_deg 20 --synthetic_prob 1.0 \
--val_synthetic_prob 1.0 --tx_range -64 64 \
--ty_range -64 64 --rot_range -15 15 --scale_range 0.85 1.15 \
--param_weight 10.0 --ncc_weight 1.0 --edge_weight 0.25 \
--charbonnier_weight 0.1 --gradient_weight 0.1 --overlap_weight 0.5 --reg_weight 0.0 \
--corr_displacement_weight 1.0 --corr_distribution_weight 0.50 \
--corr_confidence_weight 0.1 --corr_target_sigma 1.0 --student_corr_weight 1.0 \
--teacher_corr_weight 1.0 --epochs 600 --batch_size 4 --lr 0.0003 \
--weight_decay 0.00001 --grad_clip 1.0 --val_split 0.15 --split_seed 2026 \
--n_workers 8 --amp 
```

## Stage 2: real-data teacher adaptation and fine-tuning

Stage 2 may mix real unregistered rows with synthetic rows. Each training item
is synthetic with probability `synthetic_prob`. Otherwise it uses the real
unregistered stain as `moving_group`, its registered counterpart as
`target_group`, `has_params=False`, and an undefined `params_true`.

```
conda run -n reg python /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/train.py \
  --registered_root <PATH/TO/DATA> \
  --unregistered_root <PATH/TO/DATA> \
  --output_dir <PATH/TO/DIR> \
  --resume_checkpoint <PATH/TO/DIR>  \
  --best_checkpoint_name stage2_best_model.pt \
  --last_checkpoint_name stage2_last_model.pt \
  --use_teacher_branch \
  --frontend_mode raw --group_input_mode overlay \
  --affine_head_mode separated \
  --teacher_distill_weight 1.0 --detach_teacher \
  --teacher_warmup_epochs 50 \
  --no-include_group1 --force_group1_identity \
  --structural_distance_scale 0.03 \
  --structural_context_scale 0.03 \
  --structural_skeleton_radius 4 \
  --separate_group_heads --separate_group_adapters --use_group_embedding --student_fixed_adapter separate\
  --height 512 --width 512 --image_mode rgb --sfo_mode hsv \
  --crop_mode full --crop_margin 32 \
  --encoder_arch residual \
  --encoder_channels 64 96 128 192 256 \
  --encoder_blocks_per_stage 2 2 2 2 2 \
  --encoder_base_channels 24 --encoder_depth 5 \
  --feature_width 64 --correlation_feature_width 96 \
  --cost_hidden_channels 48 --cost_volume_radii 8 6 4 \
  --cost_pool_size 4 --correlation_temperature 0.07 \
  --latent_dim 384 --group_embedding_dim 32 --norm_type group \
  --model_scale_range 0.8 1.2 --translation_limit 0.5 \
  --max_rotation_deg 20 \
  --synthetic_prob 0.3 --val_synthetic_prob 0.0 \
  --tx_range -64 64 --ty_range -64 64 \
  --rot_range -15 15 --scale_range 0.85 1.15 \
  --param_weight 0.0 \
  --ncc_weight 1.0 --edge_weight 0.25 \
  --charbonnier_weight 0.1 --gradient_weight 0.1 \
  --overlap_weight 0.5 --reg_weight 0.01 \
  --corr_displacement_weight 0.0 \
  --corr_distribution_weight 0.0 \
  --corr_confidence_weight 0.0 \
  --corr_target_sigma 1.0 \
  --student_corr_weight 1.0 --teacher_corr_weight 1.0 \
  --epochs 600 --batch_size 4 --lr 0.0001 \
  --weight_decay 0.00001 --grad_clip 1.0 \
  --val_split 0.15 --split_seed 2026 \
  --n_workers 8 --amp
```



## Inference (Student model only)

Use either the full Stage-1/Stage-2 checkpoint or its derived student-only
artifact. Both load the identical student state; the teacher is never
constructed by inference. Each sample directory under
`--unregistered_root` must contain fixed Mineral stain 1 together with the
moving stains. Omit `--registered_root` entirely for target-free prediction. The
deployment call remains exactly:

```python
params = model(
    fixed_mineral=batch["fixed_mineral"],
    moving_group=batch["moving_group"],
    group=batch["group_id"],
)
```

