# Visual Backbones

The main experiment uses supervised **ViT-B/16** for ordinary visual branches.
ResNet-50 remains selectable in the same method implementations. The existing
method-specific fusion, losses, training stages, and encoder freezing decisions
are retained. This is a shared-backbone comparison, not a claim to reproduce the
original papers' numerical results.

## Run either backbone

ViT is the default for the 24 methods listed below:

```bash
python main.py --model SpotFake --dataset weibo17 --download-materials
python main.py --model SpotFake --dataset weibo17 --materials-root materials --input-mode prepared
```

For the ResNet comparison, select it when downloading and when running:

```bash
python main.py --model SpotFake --dataset weibo17 --vision-backbone resnet --download-materials
python main.py --model SpotFake --dataset weibo17 --vision-backbone resnet --materials-root materials --input-mode prepared
```

The equivalent configuration key is `model.vision_backbone: vit` or `resnet`.
The selector is unavailable for methods without an ordinary supervised visual
branch. It does not replace MAE, paired CLIP, ALBEF, region detectors, frequency
networks, or image-forensics networks.

Both choices inherit **the same current parameters for each method and dataset**:
learning rates, optimizer settings, batch sizes, losses and their weights,
dropout, training budgets, early stopping, stages, and seeds `0, 42, 2026`.
These are transferred defaults for ViT, not newly tuned ViT optima. Only the
selected visual recipe and dimensions required to connect it change.

Always evaluate a checkpoint with its own saved configuration. For a historical
ResNet configuration without `model.vision_backbone`, explicitly pass
`--vision-backbone resnet` when loading it through the current presets. Do not
load ResNet checkpoints into a ViT model or relabel previous results as ViT runs.

## Fixed weights and outputs

ViT: `google/vit-base-patch16-224`, revision
`3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3`. The checkpoint includes a trained
ImageNet-1000 classification head. RGB images are resized to a 224×224 square
with PIL bilinear interpolation, scaled by 1/255, and normalized with mean and
standard deviation `(0.5, 0.5, 0.5)`. The global representation is the final
layer-normalized CLS token; no extra ReLU, L2 normalization, or softmax is added
to the shared features. Consumers keep their own required transformations.

ResNet: the existing `resnet50` / `IMAGENET1K_V1` recipe and method-specific
preprocessing settings are retained.

Frozen ViT matrices are stored in `MMFact/MMFact`, revision
`88f12904c8db147d2fa6774c956a6afd72ac4272`, under
`datasets/<dataset>/features/image/vit_supervised/`. Matrix rows are associated
with images through `datasets/<dataset>/indexes/image_rows.jsonl`.

| Output file | Per-image shape | Meaning |
| --- | --- | --- |
| `cls768.npy` | `768` | Final CLS representation |
| `logits1000.npy` | `1000` | Raw trained ImageNet classifier logits |
| `spatial4x4.npy` | `768 × 4 × 4` | Full-image patch grid pooled from 14×14 to 4×4 |
| `patch8x8_independent.npy` | `64 × 768` | BTIC: resize the original to 640×640 with Lanczos, crop 8×8, encode each crop independently |
| `patch7x7_independent.npy` | `49 × 768` | LogicDM: resize the original to 224×224 with bilinear interpolation, crop 7×7, encode each crop independently |
| `scales224_112_64.npy` | `3 × 768` | Ordered 224, 112, 64 pixel views, with positional interpolation for the smaller views |

Independent crops are each resized to 224×224 before ViT encoding. They are
not the native patch tokens of one full-image ViT pass. Cached matrices use
FP32; online training retains the experiment's existing precision setting.

## Method bindings

“Frozen” below refers to the visual encoder, not the downstream method network.

| Methods | Ordinary visual branch with ViT |
| --- | --- |
| ADOSE, CAFE, CSFND, ERIC-FND, MRML, MVAE, SpotFake, SpotFakePlus, att-RNN | Frozen CLS768 |
| EANN | Frozen logits1000; event-adversarial training is unchanged |
| RDCM | Frozen CLS768 plus the same ViT's trained ImageNet classifier; retain method-level softmax semantic weighting |
| HMCAN | Frozen 4×4 spatial grid; retain the learned 1×1 projection and contextual attention |
| BTIC | Frozen 64 independent crop representations |
| LogicDM | Frozen 49 independent crop representations |
| MSACA | Frozen three-scale representations; ViT uses 224/112/64, ResNet retains 224/112/56 |
| ImageOnly, ConcatMLP, LateFusion | Trainable ViT CLS branch; retain each classifier/fusion and peer-selection procedure |
| CMC | Trainable image peer in the mutual-learning stage; freeze it for the later fusion stage |
| EM-FEND | Trainable full-image ViT; pool its 14×14 patch grid to 7×7, retaining CHW layout |
| FND-CLIP | Trainable ViT logits1000; the separate paired CLIP encoders remain unchanged |
| NSLM | Trainable ViT semantic branch; frequency and evidence/reasoning branches remain unchanged |
| COOLANT | Frozen ITM image branch and a separate trainable ITC image branch |
| MCAN | Trainable ViT on Weibo-17/21; frozen CLS768 on GossipCop; retain the frequency branch |

Trainable branches need original images and the pinned pretrained weights.
They cannot be replaced by fixed feature matrices without changing the training
procedure. The downloader and runtime use the same branch selection.

CSFND's visual clustering/context and CAFE/COOLANT's learned normalization are
fitted again on the training data for each new run. Existing text-side resources
remain reusable when their input recipe is unchanged. Cached features and fitted
resources from a different visual recipe must not be mixed.

MVAE reconstructs the new 768-dimensional visual target. MRML keeps its existing
projection activations, but applies the redundant raw-feature ReLU only on the
ResNet branch; signed ViT features enter its projection unchanged. Neither
adjustment changes the selected loss weights or the method's training stages.
