# Student-only correlation-volume registration

This directory provides the student-only ablation of the
`TeacherStudent` correlation-volume affine registration experiment. It uses
the same deployable student architecture, dataset, preprocessing, affine
convention, image losses, synthetic affine supervision, direct cost-volume
supervision, validation path, and inference export path as the
teacher-student experiment.

The entrypoints are deliberately thin. They call the shared implementations
in the parent `TeacherStudent` directory instead of copying the registration
model or warp code. This keeps the comparison focused on removing the teacher:

- `train.py` constructs only the student branch.
- No teacher network is allocated or saved.
- There is no teacher-only warmup.
- There is no teacher affine loss or correspondence loss.
- There is no student-teacher distillation loss.
- Registered target groups remain training-only supervision for the student's
  image-registration losses.
- Inference still requires only registered Mineral, an unregistered moving
  group, and its group ID.


## Model and loss equivalence

The student uses the raw RGB/HSV overlay frontend, group-specific moving
adapters, the shared-weight Siamese residual FPN with stage widths
`64 96 128 192 256`, projected 96-channel correlation features, three local
cost volumes with radii `8 6 4`, group embeddings, and group-specific separated
affine heads. One predicted affine is shared by every valid stain in a group.
G1 remains excluded and the Mineral/AC/Calcein identity contract remains
enabled.

For the all-synthetic Stage 1 configuration below, the student objective keeps
the same terms and weights as the student path of the teacher-student run:

- affine control-point supervision: `10.0`;
- NCC image loss: `1.0`;
- edge loss: `0.25`;
- Charbonnier loss: `0.1`;
- multiscale gradient loss: `0.1`;
- soft foreground overlap loss: `0.5`;
- expected correspondence displacement loss: `1.0`;
- correspondence-distribution loss: `0.25`;
- correspondence-confidence loss: `0.1`.

The only removed objectives are teacher-only supervision and distillation.

## Stage 1: synthetic student-only training

```bash
conda run -n reg python /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/StudentOnly/train.py \
  --registered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Registered \
  --unregistered_root /home/yec23006/projects/research/Registration/Data/Cartilage/Unregistered \
  --output_dir /home/yec23006/projects/research/Registration/Grouped/Correlation_Vol_Net/TeacherStudent/StudentOnly/ckpt/stage1_residual_corr_raw \
  --best_checkpoint_name best_model.pt \
  --last_checkpoint_name last_model.pt \
  --frontend_mode raw --group_input_mode overlay \
  --affine_head_mode separated \
  --no-include_group1 --force_group1_identity \
  --separate_group_heads --separate_group_adapters --use_group_embedding \
  --student_fixed_adapter none \
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
  --synthetic_prob 1.0 --val_synthetic_prob 1.0 \
  --tx_range -64 64 --ty_range -64 64 \
  --rot_range -15 15 --scale_range 0.85 1.15 \
  --param_weight 10.0 --ncc_weight 1.0 --edge_weight 0.25 \
  --charbonnier_weight 0.1 --gradient_weight 0.1 \
  --overlap_weight 0.5 --reg_weight 0.0 \
  --corr_displacement_weight 1.0 \
  --corr_distribution_weight 0.25 \
  --corr_confidence_weight 0.1 \
  --corr_target_sigma 1.0 \
  --student_corr_weight 1.0 \
  --epochs 600 --batch_size 4 --lr 0.0003 \
  --weight_decay 0.00001 --grad_clip 1.0 \
  --val_split 0.15 --split_seed 2026 \
  --n_workers 8 --amp --gpu_ids 0,1 \
  --wandb_project registration \
  --wandb_run_name correlation_volume_residual_direct_corr_stage1_student_only
```

Do not add `--use_teacher_branch`, `--teacher_warmup_epochs`,
`--teacher_distill_weight`, `--teacher_corr_weight`, `--detach_teacher`, or
`--freeze_teacher` to this command. The student-only entrypoint rejects
teacher-specific training options.

## Comparison budget

The command above holds the experiment length fixed at 600 epochs. In the supplied
teacher-student run, epochs 1 through 50 train only the teacher, so the student
receives 550 optimization epochs.

Report which of these comparison protocols is used:

- Equal total training duration: keep `--epochs 600`.
- Equal student optimization epochs: use `--epochs 550` for the student-only
  run.

The same `--split_seed 2026`, preprocessing, synthetic ranges, batch size, and
loss weights should be retained in either case.

## Checkpoints

Training writes:

- `best_model.pt` and `last_model.pt`: full student-only training checkpoints
  containing optimizer and scheduler state, suitable for
  `--resume_checkpoint`;
- `best_model_student.pt` and `last_model_student.pt`: compact deployable
  student artifacts suitable for inference.


Resume a student-only run from its full `.pt` checkpoint. 

## Target-free inference

Use the compact deployable checkpoint when registered target stains are not
available:

```bash
conda run -n reg python StudentOnly/infer.py \
  --checkpoint <PATH/TO/FILE> \
  --unregistered_root <PATH/TO/DIR> \
  --output_dir <PATH/TO/DIR> \
  --batch_size 4 --n_workers 8 --gpu_ids 0,1
```

Prediction uses only:

```text
registered Mineral + unregistered moving group + group ID -> affine parameters
```

Mineral is loaded from the unregistered data tree because it is the registered
fixed reference for this dataset. Registered target group stains are not
loaded.

## Inference with registered-target diagnostics

Add `--registered_root` to load registered stains for evaluation only:

```bash
conda run -n reg python StudentOnly/infer.py \
  --checkpoint <PATH/TO/FILE> \
  --registered_root <PATH/TO/DIR> \
  --unregistered_root <PATH/TO/DIR> \
  --output_dir <PATH/TO/DIR> \
  --batch_size 4 --n_workers 8 --gpu_ids 0,1
```

The registered targets are used only to produce `registered_target_metrics.csv`
with per-stain MAE and NCC. They are never passed into the model.

## Inference outputs

`infer.py` produces the same outputs as the shared TeacherStudent inference
path:

- `aligned_original_rgb/`: each aligned stain at the fixed Mineral resolution;
- `group_overlays/`: each aligned group alone and overlaid with Mineral;
- `model_space_group_overlays/`: validation-equivalent 512-by-512 overlays;
- `predicted_group_affine_parameters.csv`: the shared group affine parameters;
- `registered_target_metrics.csv`: optional per-stain MAE/NCC diagnostics when
  `--registered_root` is supplied.