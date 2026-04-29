# ScienceQA VQA — Two-Leg CNN + Text Fusion

Multi-modal multiple-choice VQA on the ScienceQA dataset.

## Architecture

```
Image ──► ResNet-50 (layer4 → [B,2048,7,7]) ──► proj [B, D]
                                                       │
                                               ┌───────┴────────┐
                                               │  Fusion module  │
                                               └───────┬────────┘
Text (Q + choice_i) ──► DistilBERT ──► CLS [B*K, D] ──┘
                                               │
                                        logits [B, K]   → argmax → answer index
```

Three fusion variants (swap via `--fusion`):

| Variant        | How it works |
|---------------|--------------|
| `early`        | concat([vis, text]) → MLP → scalar per choice |
| `late`         | separate MLP per modality, outputs summed via learned gate |
| `cross_modal`  | multi-head attention: query=text CLS, key/value=spatial CNN tokens |

Vision backbone is always ResNet-50 (non-transformer). Text encoder is DistilBERT.

## Setup

```bash
pip install -r requirements.txt --break-system-packages
```

## Training

```bash
python train.py --fusion early
python train.py --fusion late
python train.py --fusion cross_modal
```

Checkpoints are saved to `checkpoints/best_{fusion}.pt` when validation accuracy improves.

## Evaluation & submission

```bash
# Generate test-set CSV for one variant
python evaluate.py --fusion cross_modal

# Compare all three variants on validation set
python evaluate.py --compare
```

Submission format: `outputs/submission_{fusion}.csv`
```
ID,answer
2,0
5,2
6,1
```
`answer` is the predicted choice index (0-indexed), matching ScienceQA's answer field.

## Visualizations

```python
from visualize.gradcam import plot_gradcam_grid, plot_question_conditioned
from visualize.feature_space import run_analysis

# GradCAM for a list of validation samples
from data.dataset import ScienceQADataset
from transformers import AutoTokenizer
from config import Config

config    = Config(fusion_type="cross_modal")
tokenizer = AutoTokenizer.from_pretrained(config.text_model)
val_set   = ScienceQADataset("validation", config, tokenizer)
samples   = [val_set[i] for i in range(10) if val_set.data[i]["image"] is not None]

plot_gradcam_grid(model, samples, device, save_path="outputs/gradcam_grid.png")

# Same image, different questions (question-conditioned attention shift)
plot_question_conditioned(model, samples[0]["image"], [samples[0], samples[1]], device)

# Feature space analysis
run_analysis("cross_modal")
```

## Notes

- ~51% of ScienceQA questions have no image. A `has_image` gate zeros out visual
  contribution in all fusion variants when the image field is None.
- GradCAM hooks are registered on `ResNetEncoder.layer4` and fire on every forward pass.
  No model modification required to generate heatmaps post-training.
- Cross-modal attention weights `[B*K, n_heads, 1, 49]` reshape to `[7, 7]` per head
  and can be directly overlaid on the 224×224 input (each token = 32×32 px patch).
