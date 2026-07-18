# Teacher/student correlation-volume registration

This directory is an isolated teacher/student experiment built on the
deployable Correlation-Volume network. Its online frontend can use structural
descriptors, normalized raw images, or both. Run the scripts in this directory;
they import the local model, dataset, losses, and utilities.

The student is the only deployable path. It predicts one affine transform for
an acquisition group from registered Mineral, the unregistered moving stains,
and the group ID:

```python
student_params = model(fixed_mineral, moving_group, group_id)
```

It never receives a registered target stain. Normal inference therefore needs
only registered Mineral and unregistered moving stains.

During training, an independent teacher solves the easier same-stain problem:

```python
teacher_params = model.forward_teacher(target_group, moving_group, group_id)
```

`target_group` contains the registered versions of the same stains as
`moving_group`. It is allowed in this training-only teacher call and in the
loss, but it is never passed to the student's `forward()` method or required
by normal inference.

## Registration and group contract

| Group | Signals | Predicted transform |
| --- | --- | --- |
| G1 | Mineral (1), AC (2), Calcein (3) | exact identity |
| G2 | TRAP (4) | one affine |
| G3 | DAPI (5), AP (6), EDU (7) | one shared affine |
| G4 | CFO (8) | one affine |
| G5 | SFO (9) | one affine |

Every valid stain in a group is warped with the same predicted matrix. Mineral
is never warped. AC and Calcein were acquired with Mineral, so
`force_group1_identity=True` is mandatory. Training excludes G1 by default;
inference may include it to copy AC and Calcein through the identity transform.

The dataset retains the deployable eight-field contract:

| Key | Shape | Use |
| --- | --- | --- |
| `fixed_mineral` | `C x H x W` | student fixed input |
| `moving_group` | `K x C x H x W` | student/teacher moving input; `K=3` padded slots |
| `target_group` | `K x C x H x W` | training supervision and teacher fixed input only |
| `valid_group` | `K` | valid stain slots |
| `group_id` | scalar | G1-G5 routing ID |
| `stain_indices` | `K` | signal numbers; padded slots are zero |
| `params_true` | `5` | known synthetic moving-to-target affine |
| `has_params` | scalar | whether `params_true` is valid |

## Architecture

The two paths have independent learnable correlation networks and begin with
the same weights. They use the same configured frontend, but different
fixed-side sources:

```text
Deployable student
registered Mineral -> configured Mineral representation -> shared CNN/FPN --+
unregistered group -> configured group representation ---> shared CNN/FPN --+-> local cost volumes
                                                                        -> affine head
                                                                        -> student_params

Training-only teacher
registered target group -> configured group representation -> shared CNN/FPN --+
unregistered group      -> configured group representation -> shared CNN/FPN --+-> local cost volumes
                                                                            -> affine head
                                                                            -> teacher_params
```

“Shared CNN/FPN” means weight sharing between the fixed and moving sides within
one branch. Student and teacher branches themselves have independent weights.
The teacher compares registered and unregistered representations of the same
stain group, avoiding the student's more difficult Mineral-to-stain appearance
and anatomy gap.

### Frontend modes

`--frontend_mode` selects the online representation built inside both the
student and teacher:

- `structural` is the original behavior and remains the default. Mineral is
  converted to the six-channel structural descriptor
  (`torch_structural_no_enhancement_v3`) and each group is converted to a
  six-channel structural union. This mode preserves the original
  structural-model modules and checkpoint state exactly.
- `raw` skips structural conversion for prediction. The normalized model-space
  Mineral tensor is used directly, while each moving or target group uses the
  raw group representation selected by `--group_input_mode`. RGB remains RGB
  and G5 HSV remains HSV.
- `hybrid` concatenates raw channels with the same six structural channels.
  Mineral uses `raw Mineral + structural Mineral`; a moving or target group
  uses `raw group representation + structural group union`.

Raw and hybrid inputs pass through learnable Mineral-side and group-side input
adapters before the shared CNN/FPN. The adapters project different input
channel counts to the configured feature width. On the student path, the
Mineral adapter processes `fixed_mineral` and the group adapter processes
`moving_group`. On the teacher path, the same group adapter processes both
`target_group` and `moving_group`, preserving the easier same-stain comparison.
Structural mode does not insert these projection adapters.

`--group_input_mode` controls the raw part of raw/hybrid group inputs:

- `overlay` takes a channel-wise maximum over present stain slots, producing a
  valid-stain union with `C` channels.
- `stack` preserves slot separation by flattening the padded `K x C` stain
  stack into `K*C` channels. With this dataset, `K=3`; padded slots remain zero.
  Unlike the older grouped appearance model, this stack does **not** append
  presence-map channels.

Structural mode always uses its structural group union, so `group_input_mode`
does not change its image representation. It is still stored explicitly so an
experiment and its Stage 2 continuation have an unambiguous configuration.

Raw/hybrid validity is computed from the unadapted normalized images. A
conservative rectangular FOV spans the observed nonzero signal; padded and
all-zero inputs remain invalid. The mask is applied again after each learnable
adapter, so learned offsets cannot manufacture evidence in empty padding.
Hybrid validity is the union of this raw FOV and structural validity.
Structural mode retains the original descriptor-evidence gate. Consequently,
raw/hybrid matching does not depend on structural channel magnitude as its only
evidence signal.

At a fixed-grid location `x` and local displacement `d`, each selected pyramid
level computes:

```text
C(x, d) = cosine(F_fixed(x), F_moving(x + d))
```

The multiscale cost features feed the group-conditioned affine head. The
five-parameter output is `(tx, ty, rotation, scale_x, scale_y)` and is converted
to one `2 x 3` matrix for every valid stain slot.

### Affine head modes

`--affine_head_mode` selects how those five bounded geometry parameters are
predicted:

- `joint` is the original behavior and remains the default. One MLP predicts
  `(tx, ty, theta, sx, sy)` together. Its module names and tensor shapes are
  unchanged, so older joint-head TeacherStudent checkpoints strict-load.
- `separated` uses independent translation `(tx, ty)`, rotation `(theta)`, and
  scale `(sx, sy)` MLP heads. This prevents the three geometries from competing
  in one final projection while retaining the same feature encoder.
- `separated_residual` first derives a bounded, orientation-preserving coarse
  similarity from confidence-weighted cost-volume displacement moments. The
  coarse proposal has isotropic scale, so composing `coarse @ residual` with
  the separated anisotropic residual stays exactly in the existing
  five-parameter affine family and cannot introduce shear. The residual acts
  in fixed/output coordinates before the coarse fixed-to-moving map.

The closed-form coarse moments assume approximately centered, uniform valid
correspondence support. Irregular or off-center tissue can bias that proposal;
the learnable separated residual heads are responsible for correcting it.

All modes retain the configured `translation_limit`, `max_rotation_deg`, and
`model_scale_range`. Residual updates use only the remaining room on each side
of the coarse proposal, so the **final** transform cannot exceed those bounds.
The final matrix is reconstructed exactly from the returned five parameters;
`model.forward()` still returns only `B x 5`, preserving `warp_group` and the
deployment API.

With `--separate_group_heads`, every group owns the selected head. In separated
and separated-residual modes that means independent translation, rotation, and
scale heads for each group. `--no-separate_group_heads` shares one selected head
across groups. Group 1 and evidence-free pairs still return exact identity.

The code/checkpoint identifiers are:

- architecture: `correlation_volume_teacher_student_affine_v1`
- structural descriptor: `torch_structural_no_enhancement_v3`
- input contract: `fixed_mineral_moving_group_raw_v1`
- training wrapper: `TeacherStudentAffineRegistrationModel`
- deployable model: `CorrelationVolumeAffineRegistrationModel`

## Training objective

Outside teacher-only warmup, the student supplies the image-registration
prediction:

```python
student_params = model(fixed_mineral, moving_group, group_id)
# Real Stage 2 path; synthetic Stage 1 uses the composed one-pass path below.
corrected_group = warp_group(real_moving_group, student_params)
image_loss = image_objective(corrected_group, target_group, valid_group)
```

Registered targets are used only after prediction. Valid-stain and
signal-support masks remove padded slots and newly exposed affine borders. The
image objective can combine multiscale local NCC, gradient NCC, Charbonnier
distance, multiscale gradient loss, and soft foreground Dice.

When synthetic affine labels are available (`has_params=True`), the student
and enabled teacher each receive a control-point affine loss against
`params_true`. Distillation also uses control-point affine error:

```text
L = L_student_image
  + param_weight * I(has_params) * CP(student_params, params_true)
  + param_weight * I(has_params) * CP(teacher_params, params_true)
  + active_distill_weight * CP(student_params, teacher_params)
  + optional Stage-2 student regularization
```

`CP` applies both affine transforms to the four normalized image corners and
the center, then uses Smooth L1 error. This compares spatial displacement
instead of directly mixing translation, radians, and scale units.

For synthetic Stage 1, image supervision, validation images, and final
overlays avoid applying a second interpolating warp to an already transformed
and clipped tensor. Let `target_group` be the registered source before the
synthetic augmentation and let:

```text
A_syn = inverse(affine_parameters_to_matrix(params_true))
synthetic_moving = warp(target_group, A_syn)
corrected_sequential = warp(synthetic_moving, A_pred)
A_final = A_syn @ A_pred
corrected_full = warp(target_group, A_final)
```

The matrix order follows PyTorch `affine_grid`/`grid_sample`, whose affine maps
output coordinates to input coordinates. Thus `corrected_full` is the one-pass
equivalent of `corrected_sequential`, but samples from the pre-augmentation
registered tensor and avoids compounded interpolation and clipping. Synthetic
image losses, validation, and saved overlays use `corrected_full`; the
sequential result remains useful only as a diagnostic. Here the source is the
original normalized **model-space** tensor: it remains HSV for an HSV-configured
stain rather than being converted to display RGB before warping.

Real Stage 2 has no known synthetic affine or pre-augmentation registered
source for the moving observation, so its behavior is unchanged:

```python
corrected_group = warp_group(real_moving_group, student_params)
```

Synthetic train/validation reporting also includes `tx_mae_px`, `ty_mae_px`,
`theta_mae_deg`, `sx_mae`, `sy_mae`, and `control_point_error_px` separately
for student and teacher. Translation MAE follows the dataset convention of
normalizing labels by `W/2` and `H/2`; control-point error is mean Euclidean
corner/center error in the exact `align_corners=True` pixel geometry. These
metrics use only samples with `has_params=True` and are omitted during
real-only Stage 2 rather than reporting misleading zeros.

The teacher controls are:

- `--use_teacher_branch` constructs and trains the teacher. Without it, the
  experiment reduces to the student-only path.
- `--teacher_distill_weight W` multiplies the student/teacher control-point
  consistency term.
- `--detach_teacher` stops the distillation gradient at `teacher_params`, so
  distillation updates only the student. `--no-detach_teacher` lets that term
  update both branches. Detaching does not disable the teacher's synthetic
  `params_true` loss.
- `--teacher_warmup_epochs N` makes epochs satisfying `epoch <= N` genuinely
  teacher-only. The student is put in evaluation mode, its parameters and
  running state are frozen, student image/parameter losses are not optimized,
  and distillation is disabled. Only synthetic teacher control-point
  supervision against `params_true` is optimized. At `epoch > N`, the student
  is unfrozen and training switches to the normal student/image/distillation
  objective. `0` starts normal student training in epoch 1.
- `--freeze_teacher` is intended for real Stage 2 after resuming a **full**
  Stage-1 teacher/student checkpoint. It sets every teacher parameter to
  `requires_grad=False`, keeps the teacher in evaluation mode throughout
  student training, and always detaches teacher predictions used as
  distillation targets. This forced detach remains in effect even if
  `--no-detach_teacher` is also supplied.

The student learning-rate schedule begins only after teacher warmup; warmup
epochs do not consume student scheduler steps. Best-checkpoint selection uses
the teacher validation criterion during teacher-only warmup. When student
training begins, the best tracker resets and thereafter selects by the student
criterion, so a teacher-warmup score cannot prevent a later student checkpoint
from becoming best.

For real Stage 2, `has_params=False`. Use `--freeze_teacher` so the restored
teacher serves as a fixed same-group expert while the student learns from image
loss and detached distillation. A frozen Stage-2 teacher therefore requires the
full Stage-1 checkpoint containing its trained weights; a deployable
`*_student.pt` checkpoint is insufficient, and a random frozen teacher is not
meaningful.

## Checkpoints and final validation overlays

For each requested checkpoint name, training writes two artifacts:

- `<stem>.pt`: full student, teacher, optimizer, scheduler, and training state;
  use this for Stage 2 resume.
- `<stem>_student.pt`: deployable student configuration and weights only; use
  this with `inference.py`.

For example, `stage2_best_model.pt` produces the deployable
`stage2_best_model_student.pt`. `inference.py` constructs only
`CorrelationVolumeAffineRegistrationModel`; it does not construct or load the
teacher branch. Both artifacts store the canonical `student_model_config`,
including `frontend_mode`, `group_input_mode`, and the padded group-slot count.
The config also includes `affine_head_mode`; inference reconstructs that head
before strict-loading weights. Checkpoints created before this option are
interpreted as `affine_head_mode=joint`.

The full checkpoint also records literal frontend/group/head keys in its wrapper
`model_config`; the student-only artifact contains a flat deployable
`model_config` copy.

Inference has no frontend or affine-head override. It reconstructs the exact
frontend, affine head, and learnable adapter shapes from the student
checkpoint, then strict-loads the student weights. Older TeacherStudent
structural checkpoints that predate these
keys are interpreted as `frontend_mode=structural`,
`group_input_mode=overlay`, and three group slots.

After all epochs finish, training reloads the best full checkpoint and saves
validation predictions under:

```text
<output_dir>/validation_overlays/student/
<output_dir>/validation_overlays/teacher/
```

Each overlay combines registered Mineral with every valid stain warped with
one group-wide affine for that branch. Teacher overlays are diagnostic only
because they use `target_group` to make the teacher prediction. These final
overlays use the fixed sample-level validation split. Synthetic validation
overlays use the composed one-pass `warp(target_group, A_syn @ A_pred)` result;
real-data overlays use `warp(real_moving_group, A_pred)`. Files use the stable
name `{ordinal}_{sample}_G{id}.png` in both branch folders for side-by-side
review.

## Teacher transform-flow audit

`debug_teacher.py` deterministically replays the Stage 1 preprocessing pipeline
and synthetic transform ranges saved in a checkpoint. The supplied seed makes
this replay reproducible; it does not recover historical training-time random
draws because those draws were not stored. The audit requires a full
teacher/student checkpoint such as `best_model.pt`; a deployable
`*_student.pt` file does not contain the teacher weights and is rejected.

```bash
conda run -n reg python /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/debug_teacher.py \
  --checkpoint /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/ckpt/stage1_teacher_student_raw_overlay/best_model.pt \
  --registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered \
  --output_dir /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/Debug/teacher_audit \
  --max_items 8 --synthetic_seed 9090 --device cuda --gpu_ids 0
```

The audit deliberately passes an empty unregistered-data path. Its moving
source is a deterministic inverse-matrix warp of the registered target group,
not a real unregistered image; the teacher fixed source is the registered
`target_group` structural union, not Mineral. For every selected item it prints
and saves the unwarped moving source plus the true, inverse-matrix,
teacher-predicted, and student-predicted warps. It reports every image target
MAE and the teacher and student control-point errors against `params_true`.

Every image MAE uses one shared comparison support built from the union of the
target and all compared images, so one warp cannot improve its score merely by
changing the evaluated region. For G5 with `--sfo_mode hsv`, HSV tensors are
converted to visible RGB before image comparison and visualization.

The direction status is `PASS` only when the true warp beats both the unwarped
moving source and inverse-matrix warp beyond the absolute (`1e-5`) and relative
(`1e-3`) tolerances. A near tie is `INCONCLUSIVE`; contradictory evidence is
`REVIEW`. Here, `params_true` is the moving-to-target registration transform,
while its exact matrix inverse points in the opposite direction.

Each invocation creates a collision-safe
`<checkpoint>_epoch<epoch>_seed<seed>[_NNN]/` run directory under
`--output_dir`, preserving earlier audits. That run directory contains the
labeled `flow_panel.png` files, teacher fixed structural descriptors,
individual source/warp images, `metrics.csv`, and `audit.txt`.

## Stage 1: synthetic teacher/student training

Stage 1 creates `moving_group` by applying a random inverse affine to the
registered `target_group`; registered Mineral remains fixed and
`params_true` is known. Prediction still consumes the synthetic moving tensor,
but image supervision and output rendering compose the synthetic and predicted
affines and sample the registered model-space source once. With a nonzero
`teacher_warmup_epochs`, the first `N` epochs optimize only the teacher's
synthetic control-point loss; student optimization and its learning-rate
schedule begin in epoch `N+1`. Start this architecture from scratch:

```bash
conda run -n reg python /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/train.py \
  --registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered \
  --unregistered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Unregistered \
  --output_dir /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/ckpt/stage1_teacher_student \
  --best_checkpoint_name best_model.pt \
  --last_checkpoint_name last_model.pt \
  --use_teacher_branch \
  --frontend_mode structural --group_input_mode overlay \
  --affine_head_mode joint \
  --teacher_distill_weight 1.0 \
  --detach_teacher \
  --teacher_warmup_epochs 50 \
  --no-include_group1 --force_group1_identity \
  --structural_distance_scale 0.03 \
  --structural_context_scale 0.03 \
  --structural_skeleton_radius 4 \
  --separate_group_heads --separate_group_adapters --use_group_embedding \
  --height 512 --width 512 --image_mode rgb --sfo_mode hsv \
  --crop_mode full --crop_margin 32 \
  --encoder_base_channels 24 --encoder_depth 5 --feature_width 48 \
  --cost_hidden_channels 48 --cost_volume_radii 4 4 4 \
  --cost_pool_size 4 --correlation_temperature 0.07 \
  --latent_dim 384 --group_embedding_dim 32 --norm_type group \
  --model_scale_range 0.8 1.2 --translation_limit 0.5 \
  --max_rotation_deg 20 \
  --synthetic_prob 1.0 --val_synthetic_prob 1.0 \
  --tx_range -64 64 --ty_range -64 64 \
  --rot_range -15 15 --scale_range 0.85 1.15 \
  --param_weight 10.0 --ncc_weight 1.0 --edge_weight 0.25 \
  --charbonnier_weight 0.1 --gradient_weight 0.1 \
  --overlap_weight 0.5 --reg_weight 0.0 \
  --epochs 1000 --batch_size 4 --lr 0.0003 \
  --weight_decay 0.00001 --grad_clip 1.0 \
  --val_split 0.15 --split_seed 2026 \
  --n_workers 8 --amp --gpu_ids 0,1 \
  --wandb_project registration \
  --wandb_run_name correlation_volume_teacher_student_stage1_synthetic

  python /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/train.py \
  --registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered \
  --unregistered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Unregistered \
  --output_dir /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/ckpt/stage1_teacher_student_raw_overlay \
  --best_checkpoint_name best_model.pt \
  --last_checkpoint_name last_model.pt \
  --use_teacher_branch \
  --frontend_mode raw --group_input_mode overlay \
  --affine_head_mode joint \
  --teacher_distill_weight 1.0 \
  --detach_teacher \
  --teacher_warmup_epochs 80 \
  --structural_distance_scale 0.03 \
  --structural_context_scale 0.03 \
  --structural_skeleton_radius 4 \
  --separate_group_heads --separate_group_adapters --use_group_embedding \
  --height 512 --width 512 --image_mode rgb --sfo_mode hsv \
  --crop_mode full --crop_margin 32 \
  --encoder_base_channels 24 --encoder_depth 5 --feature_width 48 \
  --cost_hidden_channels 48 --cost_volume_radii 4 4 4 \
  --cost_pool_size 4 --correlation_temperature 0.07 \
  --latent_dim 384 --group_embedding_dim 32 --norm_type group \
  --model_scale_range 0.8 1.2 --translation_limit 0.5 \
  --max_rotation_deg 20 \
  --synthetic_prob 1.0 --val_synthetic_prob 1.0 \
  --tx_range -64 64 --ty_range -64 64 \
  --rot_range -15 15 --scale_range 0.85 1.15 \
  --param_weight 10.0 --ncc_weight 1.0 --edge_weight 0.25 \
  --charbonnier_weight 0.1 --gradient_weight 0.1 \
  --overlap_weight 0.5 --reg_weight 0.0 \
  --epochs 1000 --batch_size 4 --lr 0.0003 \
  --weight_decay 0.00001 --grad_clip 1.0 \
  --val_split 0.15 --split_seed 2026 \
  --n_workers 8 --amp --gpu_ids 0,1 \
  --wandb_project registration \
  --wandb_run_name correlation_volume_teacher_student_stage1_synthetic
```

The important outputs are:

```text
.../stage1_teacher_student/best_model.pt
.../stage1_teacher_student/best_model_student.pt
.../stage1_teacher_student/validation_overlays/student/
.../stage1_teacher_student/validation_overlays/teacher/
```

## Teacher-focused frontend ablations

In these runs, “teacher-only” has its literal optimization meaning during
`epoch <= teacher_warmup_epochs`: only the teacher receives updates, the
student is frozen in evaluation mode, and distillation is off. After warmup,
the student is unfrozen and the ordinary student/distillation phase begins. To
run a pure teacher diagnostic for the entire command, set
`teacher_warmup_epochs >= epochs`; such a run does not train a deployable
student. The selected `frontend_mode` applies to both branches. The following
Stage-1 commands keep the core optimization and architecture settings aligned;
their explicit warmup values determine the teacher-only prefix of each run.

### 1. Teacher structural representation

This is the original structural-union baseline. `group_input_mode=overlay` is
recorded explicitly, although structural mode builds the structural union.

```bash
conda run -n reg python /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/train.py \
  --registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered \
  --unregistered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Unregistered \
  --output_dir /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/ckpt/ablation_teacher_frontend_structural \
  --best_checkpoint_name best_model.pt \
  --last_checkpoint_name last_model.pt \
  --use_teacher_branch \
  --frontend_mode raw --group_input_mode overlay \
  --affine_head_mode separated \
  --teacher_distill_weight 1.0 \
  --detach_teacher \
  --teacher_warmup_epochs 800 \
  --no-include_group1 --force_group1_identity \
  --structural_distance_scale 0.03 \
  --structural_context_scale 0.03 \
  --structural_skeleton_radius 4 \
  --separate_group_heads --separate_group_adapters --use_group_embedding \
  --height 512 --width 512 --image_mode rgb --sfo_mode hsv \
  --crop_mode full --crop_margin 32 \
  --encoder_base_channels 24 --encoder_depth 5 --feature_width 48 \
  --cost_hidden_channels 48 --cost_volume_radii 4 4 4 \
  --cost_pool_size 4 --correlation_temperature 0.07 \
  --latent_dim 384 --group_embedding_dim 32 --norm_type group \
  --model_scale_range 0.8 1.2 --translation_limit 0.5 \
  --max_rotation_deg 20 \
  --synthetic_prob 1.0 --val_synthetic_prob 1.0 \
  --tx_range -64 64 --ty_range -64 64 \
  --rot_range -15 15 --scale_range 0.85 1.15 \
  --param_weight 10.0 --ncc_weight 1.0 --edge_weight 0.25 \
  --charbonnier_weight 0.1 --gradient_weight 0.1 \
  --overlap_weight 0.5 --reg_weight 0.0 \
  --epochs 1500 --batch_size 8 --lr 0.0003 \
  --weight_decay 0.00001 --grad_clip 1.0 \
  --val_split 0.15 --split_seed 2026 \
  --n_workers 8 --amp --gpu_ids 0,1 \
  --wandb_project registration \
  --wandb_run_name correlation_volume_teacher_frontend_structural_stage1
```

### 2. Teacher raw representation

This run gives the teacher registered and moving raw group stacks and gives the
student raw Mineral versus the raw moving group stack.

```bash
conda run -n reg python /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/train.py \
  --registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered \
  --unregistered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Unregistered \
  --output_dir /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/ckpt/ablation_teacher_frontend_raw_separate_1024 \
  --best_checkpoint_name best_model.pt \
  --last_checkpoint_name last_model.pt \
  --use_teacher_branch \
  --frontend_mode raw --group_input_mode overlay \
  --affine_head_mode separated \
  --teacher_distill_weight 1.0 \
  --detach_teacher \
  --teacher_warmup_epochs 800 \
  --no-include_group1 --force_group1_identity \
  --structural_distance_scale 0.03 \
  --structural_context_scale 0.03 \
  --structural_skeleton_radius 4 \
  --separate_group_heads --separate_group_adapters --use_group_embedding \
  --height 1024 --width 1024 --image_mode rgb --sfo_mode hsv \
  --crop_mode full --crop_margin 32 \
  --encoder_base_channels 24 --encoder_depth 5 --feature_width 48 \
  --cost_hidden_channels 48 --cost_volume_radii 4 4 4 \
  --cost_pool_size 4 --correlation_temperature 0.07 \
  --latent_dim 384 --group_embedding_dim 32 --norm_type group \
  --model_scale_range 0.8 1.2 --translation_limit 0.5 \
  --max_rotation_deg 20 \
  --synthetic_prob 1.0 --val_synthetic_prob 1.0 \
  --tx_range -64 64 --ty_range -64 64 \
  --rot_range -15 15 --scale_range 0.85 1.15 \
  --param_weight 10.0 --ncc_weight 1.0 --edge_weight 0.25 \
  --charbonnier_weight 0.1 --gradient_weight 0.1 \
  --overlap_weight 0.5 --reg_weight 0.0 \
  --epochs 1500 --batch_size 8 --lr 0.0003 \
  --weight_decay 0.00001 --grad_clip 1.0 \
  --val_split 0.15 --split_seed 2026 \
  --n_workers 8 --amp --gpu_ids 0,1 \
  --wandb_project registration \
  --wandb_run_name correlation_volume_teacher_frontend_raw_stage1_separate_1024

conda run -n reg python /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/train.py \
  --registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered \
  --unregistered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Unregistered \
  --output_dir /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/ckpt/ablation_teacher_frontend_raw_separate_1024_bigger \
  --best_checkpoint_name best_model.pt \
  --last_checkpoint_name last_model.pt \
  --use_teacher_branch \
  --frontend_mode raw --group_input_mode overlay \
  --affine_head_mode separated \
  --teacher_distill_weight 1.0 \
  --detach_teacher \
  --teacher_warmup_epochs 300 \
  --no-include_group1 --force_group1_identity \
  --structural_distance_scale 0.03 \
  --structural_context_scale 0.03 \
  --structural_skeleton_radius 4 \
  --separate_group_heads --separate_group_adapters --use_group_embedding \
  --height 1024 --width 1024 --image_mode rgb --sfo_mode hsv \
  --crop_mode full --crop_margin 32 \
  --encoder_base_channels 24 --encoder_depth 5 --feature_width 48 \
  --cost_hidden_channels 48 --cost_volume_radii 8 6 4 \
  --cost_pool_size 4 --correlation_temperature 0.07 \
  --latent_dim 384 --group_embedding_dim 32 --norm_type group \
  --model_scale_range 0.8 1.2 --translation_limit 0.5 \
  --max_rotation_deg 20 \
  --synthetic_prob 1.0 --val_synthetic_prob 1.0 \
  --tx_range -100 100 --ty_range -100 100 \
  --rot_range -20 20 --scale_range 0.85 1.15 \
  --param_weight 10.0 --ncc_weight 1.0 --edge_weight 0.25 \
  --charbonnier_weight 0.1 --gradient_weight 0.1 \
  --overlap_weight 0.5 --reg_weight 0.0 \
  --epochs 1000 --batch_size 4 --lr 0.0003 \
  --weight_decay 0.00001 --grad_clip 1.0 \
  --val_split 0.15 --split_seed 2026 \
  --n_workers 8 --amp --gpu_ids 0,1 \
  --wandb_project registration \
  --wandb_run_name correlation_volume_teacher_frontend_raw_stage1_separate_1024_bigger
```


### 3. Teacher hybrid representation

This run concatenates the corresponding raw stack and structural union before
each learnable group input adapter.

```bash
conda run -n reg python /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/train.py \
  --registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered \
  --unregistered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Unregistered \
  --output_dir /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/ckpt/ablation_teacher_frontend_hybrid \
  --best_checkpoint_name best_model.pt \
  --last_checkpoint_name last_model.pt \
  --use_teacher_branch \
  --frontend_mode hybrid --group_input_mode stack \
  --affine_head_mode joint \
  --teacher_distill_weight 1.0 \
  --detach_teacher \
  --teacher_warmup_epochs 50 \
  --no-include_group1 --force_group1_identity \
  --structural_distance_scale 0.03 \
  --structural_context_scale 0.03 \
  --structural_skeleton_radius 4 \
  --separate_group_heads --separate_group_adapters --use_group_embedding \
  --height 512 --width 512 --image_mode rgb --sfo_mode hsv \
  --crop_mode full --crop_margin 32 \
  --encoder_base_channels 24 --encoder_depth 5 --feature_width 48 \
  --cost_hidden_channels 48 --cost_volume_radii 4 4 4 \
  --cost_pool_size 4 --correlation_temperature 0.07 \
  --latent_dim 384 --group_embedding_dim 32 --norm_type group \
  --model_scale_range 0.8 1.2 --translation_limit 0.5 \
  --max_rotation_deg 20 \
  --synthetic_prob 1.0 --val_synthetic_prob 1.0 \
  --tx_range -64 64 --ty_range -64 64 \
  --rot_range -15 15 --scale_range 0.85 1.15 \
  --param_weight 10.0 --ncc_weight 1.0 --edge_weight 0.25 \
  --charbonnier_weight 0.1 --gradient_weight 0.1 \
  --overlap_weight 0.5 --reg_weight 0.0 \
  --epochs 1000 --batch_size 4 --lr 0.0003 \
  --weight_decay 0.00001 --grad_clip 1.0 \
  --val_split 0.15 --split_seed 2026 \
  --n_workers 8 --amp --gpu_ids 0,1 \
  --wandb_project registration \
  --wandb_run_name correlation_volume_teacher_frontend_hybrid_stage1
```

## Affine-head ablations

Run this complete Bash block for a controlled Stage-1 comparison. Every data,
frontend, encoder, loss, split, and optimizer setting is shared; only
`affine_head_mode`, output directory, and W&B run name change.

```bash
TEACHER_STUDENT_DIR=/home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent

COMMON_AFFINE_HEAD_ARGS=(
  --registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered
  --unregistered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Unregistered
  --best_checkpoint_name best_model.pt
  --last_checkpoint_name last_model.pt
  --use_teacher_branch
  --frontend_mode structural --group_input_mode overlay
  --teacher_distill_weight 1.0 --detach_teacher --teacher_warmup_epochs 50
  --no-include_group1 --force_group1_identity
  --structural_distance_scale 0.03 --structural_context_scale 0.03
  --structural_skeleton_radius 4
  --separate_group_heads --separate_group_adapters --use_group_embedding
  --height 512 --width 512 --image_mode rgb --sfo_mode hsv
  --crop_mode full --crop_margin 32
  --encoder_base_channels 24 --encoder_depth 5 --feature_width 48
  --cost_hidden_channels 48 --cost_volume_radii 4 4 4
  --cost_pool_size 4 --correlation_temperature 0.07
  --latent_dim 384 --group_embedding_dim 32 --norm_type group
  --model_scale_range 0.8 1.2 --translation_limit 0.5
  --max_rotation_deg 20
  --synthetic_prob 1.0 --val_synthetic_prob 1.0
  --tx_range -64 64 --ty_range -64 64
  --rot_range -15 15 --scale_range 0.85 1.15
  --param_weight 10.0 --ncc_weight 1.0 --edge_weight 0.25
  --charbonnier_weight 0.1 --gradient_weight 0.1
  --overlap_weight 0.5 --reg_weight 0.0
  --epochs 1000 --batch_size 4 --lr 0.0003
  --weight_decay 0.00001 --grad_clip 1.0
  --val_split 0.15 --split_seed 2026
  --n_workers 8 --amp --gpu_ids 0,1
  --wandb_project registration
)

# 1. Original joint head
conda run -n reg python "$TEACHER_STUDENT_DIR/train.py" \
  "${COMMON_AFFINE_HEAD_ARGS[@]}" \
  --affine_head_mode joint \
  --output_dir "$TEACHER_STUDENT_DIR/ckpt/ablation_affine_head_joint" \
  --wandb_run_name correlation_volume_affine_head_joint_stage1

# 2. Independent translation, rotation, and scale heads
conda run -n reg python "$TEACHER_STUDENT_DIR/train.py" \
  "${COMMON_AFFINE_HEAD_ARGS[@]}" \
  --affine_head_mode separated \
  --output_dir "$TEACHER_STUDENT_DIR/ckpt/ablation_affine_head_separated" \
  --wandb_run_name correlation_volume_affine_head_separated_stage1

# 3. Cost-statistics coarse similarity plus separated residual heads
conda run -n reg python "$TEACHER_STUDENT_DIR/train.py" \
  "${COMMON_AFFINE_HEAD_ARGS[@]}" \
  --affine_head_mode separated_residual \
  --output_dir "$TEACHER_STUDENT_DIR/ckpt/ablation_affine_head_separated_residual" \
  --wandb_run_name correlation_volume_affine_head_separated_residual_stage1
```

Compare the six `val_student_*` geometry metrics, `val_student_total`, and the
corresponding teacher metrics/overlays. For Stage 2, resume the full
`best_model.pt` for each run with the same `--affine_head_mode`; strict config
validation rejects accidental head changes. Inference has no head override and
rebuilds the correct deployable student from its `_student.pt` checkpoint.

## Stage 2: real unregistered fine-tuning

Stage 2 uses real unregistered stains as `moving_group`, their registered
counterparts as `target_group`, and sets `has_params=False`. Resume the full
Stage-1 checkpoint—not `best_model_student.pt`—so the trained teacher is
restored, and pass `--freeze_teacher` to make it a fixed, evaluation-mode
distillation expert. Frozen-teacher predictions are detached regardless of the
detach-teacher CLI setting. Architecture, frontend mode, group input mode,
affine head mode, descriptor, and preprocessing arguments must match Stage 1
exactly. In particular, every Stage-2 run must retain `--affine_head_mode`; a
raw or hybrid Stage-1 run must also retain its `--frontend_mode` and
`--group_input_mode`. Stage 2 directly warps each real moving group with
`A_pred`; the synthetic one-pass composition is not used.

```bash
conda run -n reg python /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/train.py \
  --registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered \
  --unregistered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Unregistered \
  --output_dir /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/ckpt/ablation_teacher_frontend_raw_separate_1024/stage2_teacher_student \
  --resume_checkpoint /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/ckpt/ablation_teacher_frontend_raw_separate_1024/best_model.pt \
  --best_checkpoint_name stage2_best_model.pt \
  --last_checkpoint_name stage2_last_model.pt \
  --use_teacher_branch \
  --frontend_mode raw --group_input_mode overlay \
  --affine_head_mode separated \
  --teacher_distill_weight 1.0 \
  --freeze_teacher --detach_teacher \
  --teacher_warmup_epochs 0 \
  --no-include_group1 --force_group1_identity \
  --structural_distance_scale 0.03 \
  --structural_context_scale 0.03 \
  --structural_skeleton_radius 4 \
  --separate_group_heads --separate_group_adapters --use_group_embedding \
  --height 1024 --width 1024 --image_mode rgb --sfo_mode hsv \
  --crop_mode full --crop_margin 32 \
  --encoder_base_channels 24 --encoder_depth 5 --feature_width 48 \
  --cost_hidden_channels 48 --cost_volume_radii 4 4 4 \
  --cost_pool_size 4 --correlation_temperature 0.07 \
  --latent_dim 384 --group_embedding_dim 32 --norm_type group \
  --model_scale_range 0.8 1.2 --translation_limit 0.5 \
  --max_rotation_deg 20 \
  --synthetic_prob 0.0 --val_synthetic_prob 0.0 \
  --tx_range -64 64 --ty_range -64 64 \
  --rot_range -15 15 --scale_range 0.85 1.15 \
  --param_weight 0.0 --ncc_weight 1.0 --edge_weight 0.25 \
  --charbonnier_weight 0.1 --gradient_weight 0.1 \
  --overlap_weight 0.5 --reg_weight 0.01 \
  --epochs 800 --batch_size 4 --lr 0.00001 \
  --weight_decay 0.00001 --grad_clip 1.0 \
  --val_split 0.15 --split_seed 2026 \
  --n_workers 8 --amp --gpu_ids 0,1 \
  --wandb_project registration \
  --wandb_run_name correlation_volume_teacher_student_stage2_real
```

## Target-free student inference

Use the derived student-only Stage-2 artifact. The registered root may contain
only Mineral; registered group stains are not discovered or loaded. The
deployment call remains exactly:

```python
params = model(fixed_mineral, moving_group, group_id)
```

Neither `target_group` nor a frontend CLI argument is required for prediction.

```bash
conda run -n reg python /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/inference.py \
  --checkpoint /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/ckpt/stage2_teacher_student/stage2_best_model_student.pt \
  --registered_root /home/yec23006/projects/research/Registration/Data/Testdata/Registered \
  --unregistered_root /home/yec23006/projects/research/Registration/Data/Testdata/Unregistered \
  --output_dir /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/Results/stage2_student \
  --include_group1 \
  --batch_size 4 --n_workers 4 --device cuda --gpu_ids 0,1
```

Outputs include aligned original RGB stains,
`predicted_group_affine_parameters.csv`, group-only overlays, and
`group_overlays/<sample>/groupN_with_mineral_overlay.png`.

## Optional registered-target evaluation

Add `--eval_with_registered_targets` only when registered counterparts exist
for all moving stains. The student prediction is unchanged: targets are used
only after prediction to compute support-masked MAE/NCC.

```bash
conda run -n reg python /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/inference.py \
  --checkpoint /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/ckpt/stage2_teacher_student/stage2_best_model_student.pt \
  --registered_root /home/yec23006/projects/research/Registration/Data/Testdata/Registered \
  --unregistered_root /home/yec23006/projects/research/Registration/Data/Testdata/Unregistered \
  --output_dir /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/Results/stage2_student_registered_target_eval \
  --include_group1 --eval_with_registered_targets \
  --batch_size 4 --n_workers 4 --device cuda --gpu_ids 0,1
```

This additionally writes `registered_target_metrics.csv` and prints the mean
MAE and NCC. Registered targets remain absent from model prediction.

## Compatibility and smoke checks

This experiment intentionally rejects original Correlation-Volume checkpoints
with a different architecture tag. Train Stage 1 in this directory before
running Stage 2. Frozen-teacher Stage 2 requires its full teacher/student
checkpoint, while inference should use the corresponding `_student.pt`
artifact. These training-flow changes do not alter the dataset fields, model
forward signature, group-shared affine rule, `warp_group` API, or target-free
deployment contract.

```bash
cd /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent
conda run -n reg python smoke_test.py --device cpu
```

For a bounded training check, append
`--max_train_items 1 --max_val_items 1`. For bounded inference, append
`--max_items 1`.
