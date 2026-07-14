# Grouped Fluorescence Registration

## Project contract and paper methods reference

This file is both the implementation contract for the repository and an implementation-faithful methods reference for the paper. When code and prose disagree, inspect the code, correct the disagreement, and update this file. Do not describe an optional or merely implemented technique as part of the active method.

## 1. Registration problem

Nine fluorescence stainings are available from the same tissue. Mineral defines the fixed coordinate system. Every other staining is a moving signal whose affine transform onto the Mineral coordinate system must be estimated.

| Index | Signal | Role | Acquisition group |
|---:|---|---|---:|
| 1 | Mineral | Fixed reference | 1 |
| 2 | AC | Moving | 1 |
| 3 | Calcein | Moving | 1 |
| 4 | TRAP | Moving | 2 |
| 5 | DAPI | Moving | 3 |
| 6 | AP | Moving | 3 |
| 7 | EDU | Moving | 3 |
| 8 | CFO | Moving | 4 |
| 9 | SFO | Moving | 5 |

The moving members used by the dataset are therefore

\[
G_1=\{\mathrm{AC},\mathrm{Calcein}\},\quad
G_2=\{\mathrm{TRAP}\},\quad
G_3=\{\mathrm{DAPI},\mathrm{AP},\mathrm{EDU}\},\quad
G_4=\{\mathrm{CFO}\},\quad
G_5=\{\mathrm{SFO}\}.
\]

Mineral is conceptually associated with Group 1, but it is not inserted into the Group-1 moving stack. It is always supplied through a separate fixed-image branch.

The central invariant is:

> One tissue sample and one acquisition group produce exactly one affine parameter vector. That same affine transform is applied to every valid signal in the group.

The registered version of each moving stain is already in the Mineral coordinate system and is used as that stain's training target. Mineral conditions the network and defines the fixed frame; the image losses compare a warped moving stain with the registered target of the same stain, not directly with Mineral.

## 2. Active reference configuration

The current Stage-2 checkpoint is

`/home/yec23006/projects/research/Registration/Grouped/ckpt/group_stack_fixed/stage2_best_model.pt`.

Its saved metadata was checked on 2026-07-14 and contains the following active configuration. Use checkpoint metadata and the explicit Stage-1/Stage-2 commands at the top of `train.py`, not the smaller argument-parser defaults, when describing the paper method.

| Component | Active setting |
|---|---|
| Input size | 1024 x 1024 |
| Image representation | RGB for every signal, including SFO |
| Group representation | Three fixed RGB slots plus three presence maps: 12 channels |
| Fixed input | RGB Mineral: 3 channels |
| Crop | Full image, aspect-preserving resize, centered zero padding |
| Encoder | Five-scale dual stream, base width 48, intermediate fusion |
| Encoder widths | 48, 96, 192, 384, 768 |
| Normalization | GroupNorm |
| Coordinate channels | Enabled |
| Latent dimension | 384 |
| Group embedding | 32 dimensions |
| Group specialization | Five residual input adapters and five affine heads |
| Spatial pooling | Adaptive average pooling to 4 x 4 |
| Affine bounds | translation +/-0.5 normalized units, rotation +/-20 degrees, scales [0.8, 1.2] |
| Group-1 identity | Not forced |
| Trainable parameters | 29,790,557 |

The parameter breakdown is 29,077,680 in the dual-stream encoder, 705,305 across the five affine heads, 7,380 across the five residual adapters, and 192 in the group embedding.

## 3. End-to-end method

```text
moving signals for one sample/group
        |
        +-- fixed slot packing + presence maps --> group-specific enhancement
        |                                           |
        |                                  group-specific residual adapter
        |                                           |
Mineral --> fixed RGB branch ---------------- dual-stream encoder
                                                    |
                                      multi-scale comparison fusion
                                                    |
                                  pooled latent + group-ID embedding
                                                    |
                                        group-specific affine head
                                                    |
                              one (tx, ty, theta, sx, sy) prediction
                                                    |
                           repeat the same matrix across all group slots
                                                    |
                     warp unenhanced moving signals and compute losses
```

There is no image decoder. The model is a convolutional affine regressor followed by a differentiable spatial transformer. In paper text, call the output modules **group-specific affine regression heads**, not decoders.

## 4. Dataset construction and group tensors

`dataset.py` pairs registered and unregistered sample directories using a base identifier formed by removing `_org` or `_aligned`. Stain indices are parsed from filenames. A grouped dataset item is a `(sample, group)` pair.

A group item is retained when at least one registered target is present. In real-data mode it must also have at least one corresponding unregistered signal. Groups have different numbers of signals, so every group is padded to the maximum of three slots:

- Group 1 uses the ordered slots AC, Calcein, padding.
- Group 2 uses TRAP, padding, padding.
- Group 3 uses DAPI, AP, EDU.
- Group 4 uses CFO, padding, padding.
- Group 5 uses SFO, padding, padding.

Missing signals and padded slots are zero-filled. A Boolean validity mask prevents them from contributing to any loss or output. Stain indices are retained so inference routes each prediction back to the corresponding source image.

With active RGB `stack` mode, three RGB slots contribute nine channels and three full-image binary presence maps contribute another three channels:

\[
I_g\in\mathbb{R}^{B\times12\times1024\times1024},\qquad
I_f\in\mathbb{R}^{B\times3\times1024\times1024}.
\]

The presence maps indicate whether an entire stain slot exists; they are not foreground or tissue masks. The unenhanced moving and target tensors are retained separately with shape

\[
M,T\in\mathbb{R}^{B\times3\times3\times1024\times1024}.
\]

The first dimension after the batch is the stain-slot dimension. This separation is essential: `group_input` is used to estimate the transform, while `moving_group` and `target_group` are used for differentiable warping and loss calculation.

## 5. Geometric and intensity preprocessing

Images are loaded as floating-point RGB arrays in `[0,1]`. The fixed Mineral image is additionally converted to grayscale and Otsu-thresholded to create a Mineral mask. A single geometry derived from the fixed canvas is reused for all signals in a sample:

1. select either the full image or a Mineral bounding box with a margin;
2. resize bilinearly while preserving aspect ratio;
3. center the result on a zero-padded model canvas;
4. normalize every output channel independently with a z-score.

The active protocol uses `crop_mode=full`, so no image content is cropped. If an image's original dimensions differ from Mineral, it is first resized to the Mineral canvas before the common letterbox transform. In the active full-image protocol, the Mineral mask is not used to weight the loss.

RGB resizing is performed before any optional SFO color-space conversion. This avoids interpolating hue, which is circular. The active checkpoint uses RGB SFO. When `sfo_mode=hsv` is selected experimentally, SFO remains a three-channel HSV encoder representation; grayscale is not part of the active protocol.

### 5.1 Encoder-only structural enhancement

Enhancement is applied only to `group_input`. It never alters the unenhanced moving image used by the spatial transformer, the registered target used by the loss, or the original RGB image saved by inference. Consequently, enhancement helps the regressor detect corresponding structure without changing the registration objective or final image content.

The common structural operator for an intensity channel `x` is:

1. CLAHE with clip limit 2.0 and an 8 x 8 tile grid;
2. Otsu threshold `T`;
3. soft foreground gate

\[
g(x)=\frac{1}{1+\exp(-(255x-T)/12)};
\]

4. 3 x 3 Sobel derivatives and a gradient magnitude normalized by its maximum;
5. structural response

\[
S(x)=\operatorname{clip}(\operatorname{CLAHE}(x)g(x)+0.35E(x),0,1).
\]

The final enhanced encoder input is always a convex blend

\[
I_{\mathrm{enc}}=(1-\alpha)I+\alpha I_{\mathrm{enh}}.
\]

Only Groups 2, 4, and 5 are enhanced in the active method.

#### TRAP / Group 2

- Compute per-pixel maximum RGB intensity.
- Use Otsu thresholding to obtain foreground.
- Apply one dilation followed by one erosion with a 3 x 3 elliptical kernel. This is a morphological closing that connects nearby signal components.
- Compute the common contrast/edge response on the maximum intensity.
- Inside the connected mask, use `max(S(I) M, 0.65 M)` and repeat the scalar structure over the three RGB channels.
- Blend with strength `alpha=0.8`.

#### CFO / Group 4

- Apply the common CLAHE, soft Otsu foreground, and Sobel operator independently to the three RGB channels.
- Blend the structural RGB representation with strength `alpha=0.5`.

#### SFO / Group 5

- Convert RGB to HSV internally for selection only.
- Retain pixels with hue from 70 to 200 degrees inclusive, saturation at least 0.18, and value at least the Otsu value threshold.
- Apply a 3 x 3 elliptical opening and then closing to remove isolated pixels and fill small gaps.
- Apply the common contrast/edge response to the selected value channel.
- Convert the selected representation back to RGB.
- Use `alpha=1.0`; excluded colors are therefore suppressed in the encoder input.

The externally visible representation is still RGB when `sfo_mode=rgb`. HSV is an internal selection device, not a requirement on the group tensor.

In synthetic training, the raw registered RGB canvas is displaced first. SFO conversion and group enhancement are applied only after that warp, so HSV hue is never spatially interpolated and enhancement matches the displaced encoder input.

## 6. Model structure

### 6.1 Group-specific residual input adapters

Before the shared encoder, group ID `g` selects adapter `g-1`. Each of the five adapters is

```text
12-channel input
 -> 3x3 Conv(12, 12, bias=False)
 -> GroupNorm(6 groups)
 -> GELU
 -> 1x1 Conv(12, 12)
 -> residual addition to the original input
```

The final 1 x 1 convolution is initialized with zero weights and zero bias, making every adapter exactly the identity at initialization. The adapters can then learn signal-specific preprocessing without forcing the common encoder to represent every fluorescence appearance identically. Mineral does not use an adapter.

### 6.2 Dual-stream convolutional encoder

This is not a weight-sharing Siamese network: the group and Mineral branches have independent convolutional parameters.

Normalized x/y coordinate planes in `[-1,1]` are concatenated to both inputs. The first group stage therefore receives 14 channels and the first fixed stage receives 5 channels. Coordinate channels preserve absolute spatial information that would otherwise be weakened by convolution and global pooling.

Each branch contains five convolutional stages. A stage consists of

```text
3x3 stride-2 Conv -> GroupNorm -> GELU
3x3 stride-1 Conv -> GroupNorm -> GELU
```

For the 1024 x 1024 reference configuration, feature shapes are:

| Scale | Group and fixed feature shape | Fusion input channels |
|---:|---|---:|
| 1 | `B x 48 x 512 x 512` | 192 |
| 2 | `B x 96 x 256 x 256` | 384 |
| 3 | `B x 192 x 128 x 128` | 768 |
| 4 | `B x 384 x 64 x 64` | 1,536 |
| 5 | `B x 768 x 32 x 32` | 3,072 |

At every scale `l`, the group feature `G_l` and fixed Mineral feature `F_l` are combined through direct features, absolute differences, and multiplicative interactions:

\[
H_l=\phi_l([G_l,F_l,|G_l-F_l|,G_l\odot F_l]),
\]

where `phi_l` is a bias-free 1 x 1 convolution followed by GroupNorm and GELU. The fusion output is added to both branches before the next scale:

\[
G_l\leftarrow G_l+H_l,\qquad F_l\leftarrow F_l+H_l.
\]

The final fused feature is adaptively average-pooled to 4 x 4, flattened from 12,288 values, and projected through `Linear(12288,384)`, LayerNorm, and GELU to a 384-dimensional latent vector. No dropout is active.

### 6.3 Group embedding and separated affine heads

A learned 32-dimensional embedding of the integer group ID is concatenated with the 384-dimensional image latent, giving a 416-dimensional regressor input. The embedding table has six entries; IDs 1 through 5 are used.

There are five independent heads, with explicit routing by group ID:

```text
416 -> Linear(256) -> LayerNorm -> GELU
    -> Linear(128) -> LayerNorm -> GELU
    -> Linear(5)
```

Each head's final layer is initialized to zero. Group-specific batches are gathered, processed by the corresponding adapter/head, and restored to their original batch order. Thus the code does not mix a Group-2 latent with a Group-4 or Group-5 head. Multiple stains within Group 1 or Group 3 do not receive separate heads; they intentionally share the one head output for their acquisition group.

The five outputs are

\[
p=(t_x,t_y,\theta,s_x,s_y).
\]

For raw head output `z`, the bounded parameterization is

\[
t_x=0.5\tanh z_1,\quad t_y=0.5\tanh z_2,
\]

\[
\theta=(20\text{ degrees in radians})\tanh z_3,
\]

\[
s_x=0.8+0.4\sigma(z_4),\quad s_y=0.8+0.4\sigma(z_5).
\]

Because the output layer is zero-initialized and the scale interval is symmetric around 1, the initial prediction is exactly `(0,0,0,1,1)`.

### 6.4 Affine matrix and spatial-transformer convention

Parameters are converted to the five-degree-of-freedom matrix

\[
A(p)=
\begin{bmatrix}
s_x\cos\theta & -s_y\sin\theta & t_x\\
s_x\sin\theta &  s_y\cos\theta & t_y
\end{bmatrix}.
\]

The transform supports translation, rotation, and independent x/y scales. It does not include shear.

PyTorch `affine_grid` uses an output-to-input convention. Therefore `A(p)` maps fixed/output normalized coordinates to moving/input normalized coordinates. `grid_sample` then produces the moving image in the fixed frame using bilinear interpolation, zero padding, and `align_corners=True`.

For every group item, `A(p)` is expanded across all three slots before warping. This is the code-level guarantee that all valid signals from the same acquisition group receive the same transformation.

At 1024 x 1024, a normalized translation bound of 0.5 corresponds to approximately 255.75 pixels under `align_corners=True`.

## 7. Active loss functions

### 7.1 Notation and loss routing

For sample/group item `b` and valid stain slot `k`, let

- `M_bk` be the unenhanced moving stain;
- `T_bk` be the registered target of the same stain;
- `p_b` be the single predicted parameter vector for the group;
- `W_bk = grid_sample(M_bk, A(p_b))` be the warped moving stain.

Define the valid set

\[
\mathcal{D}=\{(b,k):\text{valid_group}_{bk}=1\}
\]

and the sparse-signal set

\[
\mathcal{S}=\{(b,k)\in\mathcal{D}:g_b\in\{2,4,5\}\}.
\]

The multi-scale local NCC, gradient-NCC, and Charbonnier terms use every valid stain in Groups 1 through 5. The multi-scale gradient and soft foreground overlap terms use only TRAP, CFO, and SFO. Padded or missing slots never contribute. Enhancements never enter the loss tensors.

All names ending in `_ncc` in the logs are losses of the form `1 - correlation`; lower is better.

### 7.2 Multi-scale local normalized cross-correlation

At each scale `s` in `{1,2,4}`, images are unchanged for `s=1` or average-pooled with kernel and stride `s`. A 9 x 9 average-pooling window with stride 1 and padding 4 gives local statistics

\[
\mu_W=E[W],\quad \mu_T=E[T],
\]

\[
v_W=\max(E[W^2]-\mu_W^2,0),\quad
v_T=\max(E[T^2]-\mu_T^2,0),
\]

\[
c=E[WT]-\mu_W\mu_T,\qquad
\rho=\frac{c}{\sqrt{v_Wv_T+10^{-5}}}.
\]

Only locations satisfying `v_W + v_T > 1e-5` are informative. The scale loss and equal-weight multi-scale loss are

\[
L_{LNCC}^{(s)}=1-\operatorname{mean}(\rho\;\text{over informative entries}),
\]

\[
L_{MS-LNCC}=\frac{1}{3}\sum_{s\in\{1,2,4\}}L_{LNCC}^{(s)}.
\]

If a scale has no informative entry, its loss is 1. This dense structural term supplies coarse-to-fine capture while ignoring flat fluorescence regions.

### 7.3 Gradient normalized cross-correlation

For each channel, forward differences are computed in x and y. The last row/column is replicate-padded, and the gradient magnitude is

\[
G(X)=\sqrt{(D_xX)^2+(D_yX)^2+10^{-8}}.
\]

A global spatial NCC is calculated independently for every image/channel and clamped to `[-1,1]`:

\[
L_{edge}=1-\operatorname{mean}_{i,c}
\frac{\operatorname{Cov}(G(W),G(T))}
{\sqrt{\operatorname{Var}(G(W))\operatorname{Var}(G(T))}+10^{-8}}.
\]

This loss uses first-order finite differences. It is not a Sobel loss; Sobel filtering is used only in input enhancement.

### 7.4 Charbonnier photometric loss

The robust photometric term is

\[
L_{char}=\operatorname{mean}_{\mathcal{D},c,x,y}
\sqrt{(W-T)^2+(10^{-3})^2}.
\]

The Charbonnier penalty reduces the influence of stain artifacts relative to a squared-error loss.

### 7.5 Multi-scale gradient discrepancy for sparse signals

This term is evaluated only on `\mathcal{S}`. At scales `{1,2,4}`, forward x/y differences are computed, with the final row/column zero-padded in this loss. For direction `d` in `{x,y}`,

\[
L_{s,d}=\operatorname{mean}\sqrt{(D_dW_s-D_dT_s)^2+(10^{-3})^2},
\]

\[
L_{grad}=\frac{1}{6}\sum_{s\in\{1,2,4\}}\sum_{d\in\{x,y\}}L_{s,d}.
\]

This term penalizes displaced sparse edges across coarse and fine resolutions.

### 7.6 Soft foreground Dice for sparse signals

For a z-score-normalized image `X`, form a nonnegative channel-max response and normalize it by its spatial maximum:

\[
r_X=\frac{\max_c\operatorname{ReLU}(X_c)}
{\max_{x,y,c}\operatorname{ReLU}(X_c)+10^{-6}}.
\]

The differentiable foreground mask is

\[
F_X=\sigma(12(r_X-0.35)).
\]

For TRAP, CFO, and SFO,

\[
L_{overlap}=1-\operatorname{mean}_i
\frac{2\sum F_WF_T+10^{-6}}
{\sum F_W+\sum F_T+10^{-6}}.
\]

This aligns sparse foreground support without introducing a hard, non-differentiable threshold into training.

### 7.7 Synthetic affine control-point supervision

Synthetic Stage 1 provides a known registration transform `p*`. Rather than applying MSE directly to quantities with incompatible units, the predicted and true transforms are compared at four normalized image corners and the center:

\[
\mathcal{P}=\{(-1,-1,1),(1,-1,1),(-1,1,1),(1,1,1),(0,0,1)\}.
\]

For every point and coordinate, define

\[
e=A(p)q-A(p^*)q.
\]

The supervision is Smooth L1 with `beta=0.02`, averaged over samples, points, and x/y coordinates:

\[
L_{CP}=\operatorname{mean}\begin{cases}
e^2/(2\beta), & |e|<\beta,\\
|e|-\beta/2, & \text{otherwise}.
\end{cases}
\]

Synthetic moving images are generated with `A(p*)^{-1}`, so `A(p*)` is the correct moving-to-target registration transform.

### 7.8 Identity regularization for real data

Samples without a ground-truth affine label use a small identity prior:

\[
L_{reg}=\operatorname{mean}(p-(0,0,0,1,1))^2.
\]

This operates directly on the five raw parameter dimensions and is used only at low weight in Stage 2.

### 7.9 Stage-specific total objectives

| Loss | Applies to | Stage 1 weight | Stage 2 weight |
|---|---|---:|---:|
| Multi-scale local NCC | All valid stains | 1.00 | 1.00 |
| Gradient NCC | All valid stains | 0.25 | 0.25 |
| Charbonnier | All valid stains | 0.10 | 0.10 |
| Multi-scale gradient | TRAP, CFO, SFO | 0.10 | 0.10 |
| Soft foreground Dice | TRAP, CFO, SFO | 0.50 | 0.50 |
| Affine control points | Synthetic group items | 10.00 | 0.00 |
| Identity regularization | Real group items | 0.00 | 0.01 |

Thus the documented Stage-1 objective is

\[
L_1=L_{MS-LNCC}+0.25L_{edge}+0.10L_{char}+0.10L_{grad}
+0.50L_{overlap}+10L_{CP},
\]

and the documented Stage-2 objective is

\[
L_2=L_{MS-LNCC}+0.25L_{edge}+0.10L_{char}+0.10L_{grad}
+0.50L_{overlap}+0.01L_{reg}.
\]

In a mixed synthetic/real batch, control-point supervision is evaluated only for rows with known synthetic parameters, while identity regularization is evaluated only for real rows. A zero-weight term is omitted, and a sparse term is omitted for a batch containing no Group-2, Group-4, or Group-5 item.

Multi-stain Groups 1 and 3 contribute one dense-loss image per valid slot, whereas single-stain Groups 2, 4, and 5 contribute one. The affine parameter loss and regularizer operate once per group item.

## 8. Two-stage training protocol

### 8.1 Stage 1: synthetic supervised warm-start

Every registered target is synthetically displaced (`synthetic_prob=1.0` for training and validation). One transform is sampled per group item and applied consistently to all its valid stains.

The active sampling distribution is uniform:

\[
t_x,t_y\sim U(-64,64)\text{ model-space pixels},
\]

\[
\theta\sim U(-15,15)\text{ degrees},\qquad s\sim U(0.85,1.15).
\]

Pixel translations are divided by `W/2` and `H/2` to obtain normalized affine-grid coordinates. Synthetic scale is isotropic (`s_x=s_y=s`) even though the model can predict independent scales. Stage 2 can adapt those scale outputs independently.

The current documented Stage-1 command initializes model weights from `ckpt/group_stack_fixed/best_model.pt`. The option is named `resume_checkpoint`, but it loads only model weights; it does not restore optimizer state, scheduler state, epoch number, or best score. A legacy shared head is copied into all five heads, and missing group adapters are initialized as identities when checkpoint migration is needed.

Stage-1 optimization uses 400 configured epochs, batch size 8, AdamW with learning rate `3e-4` and weight decay `1e-5`, cosine annealing to 5% of the initial learning rate, global gradient-norm clipping at 1.0, CUDA automatic mixed precision with initial scale 4096, and GPU IDs 0 and 1 through `DataParallel`. The documented run uses eight persistent data workers and W&B project `registration`, run name `group_stack_affine_stage1_rgb`.

### 8.2 Stage 2: real paired-data fine-tuning

Stage 2 uses the real unregistered images (`synthetic_prob=0.0` for training and validation) and initializes model weights from the Stage-1 best checkpoint. It has no ground-truth affine parameter labels, but it is not unpaired or target-free: the corresponding registered same-stain images remain the targets for all image losses.

The documented Stage-2 run uses 400 configured epochs, batch size 8, AdamW with learning rate `1e-5` and weight decay `1e-5`, the same cosine schedule, gradient clipping, AMP, two-GPU `DataParallel`, and data-loading settings. It logs to W&B project `registration`, run name `group_stack_affine_stage2_rgb`.

The checkpoint `stage2_best_model.pt` currently records epoch 217. Do not state that all 400 epochs completed unless the final run history or a later checkpoint verifies it.

### 8.3 Train/validation split

Training and validation are split by tissue sample, not by group item, preventing signals or acquisition groups from the same tissue from leaking across partitions. The active validation fraction is 0.15 with split seed 2026. The current saved split contains 14 training samples/70 group items and 2 validation samples/10 group items; recheck the saved metadata if the dataset changes.

Stage-1 validation synthesis is deterministic with seed `9090 + 1009 * item_index`. Training synthesis creates an unseeded `numpy.random.default_rng()` for each access. Python, legacy NumPy, and PyTorch are seeded, but Stage-1 training augmentation is therefore not exactly reproducible from `--seed` alone. Do not claim strict deterministic reproducibility without changing this behavior.

### 8.4 Validation, optimization safety, and checkpointing

Validation recomputes the stage-specific total objective without gradients and separately reports multi-scale local-NCC loss for Groups 1 through 5. Metrics named `val_group{g}_ncc` are `1-NCC` losses, so smaller values indicate better alignment.

Training logs each applicable component, total loss, optimizer-step count, skipped AMP-step count, epoch, and all per-group validation NCC losses to W&B. Non-finite total losses abort training. Gradients are unscaled before norm clipping; AMP may skip a non-finite update, and training aborts if every update in an epoch is skipped.

The best checkpoint is selected by minimum `val_total`, not by landmarks, target-registration error, or an external test metric. The last checkpoint is overwritten every epoch. Checkpoints store model, optimizer, scheduler, epoch, metrics, model configuration, sample split, and preprocessing configuration. They do not store every loss/optimizer CLI argument, so retain the exact command and W&B configuration for paper reproducibility.

There is no early stopping and no dedicated test split in `train.py`.

## 9. Inference

`inference.py` reconstructs preprocessing and model architecture from checkpoint metadata. It predicts one parameter vector per sample/group and reuses it for all valid group signals.

The normalized output-to-input PyTorch matrix is converted to pixel coordinates, conjugated by the preprocessing transform, and adjusted when the moving source resolution differs from the fixed Mineral resolution. Original unregistered RGB images are then warped directly at Mineral's original resolution with bilinear interpolation, OpenCV `WARP_INVERSE_MAP`, and constant black padding.

Inference saves:

- each aligned original-resolution RGB signal;
- a channel-wise maximum overlay for every acquisition group;
- a CSV containing sample, group, stain index, normalized translations, rotation in degrees, independent x/y scales, and output path.

Current inference still requires `registered_root`, including registered non-Mineral targets, because dataset construction uses them to discover valid group members. Those target tensors are not used to make the prediction. Do not claim that inference needs only Mineral plus unregistered stains unless this dependency is refactored and tested.

## 10. Implemented but not active in the documented objective

`losses.py` also defines the following functions, but `train.py::grouped_loss` does not add them to the current Stage-1 or Stage-2 objective:

- standalone SSD;
- intensity global NCC, except that its implementation is used internally after gradient-magnitude conversion for active gradient NCC;
- differentiable mutual information;
- correlation ratio;
- generic soft Dice;
- soft Jaccard distance;
- raw weighted affine-parameter MSE;
- biological/anatomical regional prior.

Mineral, boundary, and exterior masks do not weight the current losses. `compute_boundary_mask` and `compute_exterior_mask` are currently unused. Do not claim these techniques as training contributions unless they are connected to `grouped_loss`, enabled in the actual experiment, and recorded in checkpoint/run metadata.

Optional code paths that are not part of the active reference checkpoint include `single` and `overlay` group inputs, grayscale images, HSV SFO input, Mineral-bounding-box cropping, early concatenation fusion, BatchNorm/InstanceNorm, disabled CoordConv, a shared affine head, disabled adapters, and forced Group-1 identity.

## 11. Paper wording and reporting requirements

Use these descriptions:

- **Model:** dual-stream, multi-scale convolutional affine regressor with intermediate comparison fusion, CoordConv, group-specific residual adapters, a group embedding, and five group-specific affine heads.
- **Transformation:** five-degree-of-freedom affine transformation with translation, rotation, and independent axis scaling; no shear.
- **Sharing:** one predicted transform is shared by all signals in the same acquisition group.
- **Supervision:** synthetic affine control-point supervision in Stage 1 and paired same-stain image supervision without affine labels in Stage 2.
- **Enhancement:** encoder-only, group-specific RGB structural enhancement; SFO uses internal HSV selection and returns to RGB.
- **Validation:** sample-level held-out validation and minimum total validation loss for model selection.

Avoid these inaccurate descriptions:

- encoder-decoder or separated decoders;
- weight-sharing Siamese encoder;
- direct multimodal loss between a moving stain and Mineral;
- unpaired or unsupervised Stage-2 training;
- Sobel gradient loss;
- shear prediction;
- mask-weighted, mutual-information, biological-prior, generic Dice, or Jaccard training;
- forced identity for Group 1;
- fully deterministic Stage-1 augmentation;
- completion of 400 Stage-2 epochs without checking the final run;
- parser defaults as the experimental configuration.

Before reporting results, record the final checkpoint path and epoch, W&B run ID, exact CLI, sample IDs in each split, code revision, final loss weights, and the metric used for checkpoint selection. The current code does not compute target registration error or landmark error; such a metric must be implemented and evaluated before it can be reported.

## 12. Source map and maintenance rules

- `dataset.py`: signal/group mapping, file pairing, group slots, validity, synthetic displacement, and enhanced encoder inputs.
- `models.py`: adapters, dual-stream encoder, feature fusion, embedding, affine heads, and group routing.
- `utils.py`: image preprocessing, enhancement, affine convention, spatial transform, and normalized/pixel matrix conversion.
- `losses.py`: loss equations.
- `train.py`: active objective routing, two-stage commands, sample-level split, optimization, validation, W&B, and checkpoints.
- `inference.py`: checkpoint reconstruction, original-resolution transformation, overlays, and CSV output.

Before modifying a function, search the entire repository for every definition, import, and call site. Read every affected file before editing. Preserve the group-sharing invariant and the output-to-input affine convention. Run syntax checks, CLI help checks, focused smoke tests, and the relevant Stage-1/Stage-2 or inference command after changes. Update this methods reference whenever an active model, preprocessing, loss, training, validation, or inference technique changes.
