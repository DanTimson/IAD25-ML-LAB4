# ScienceQA VQA

Visual Question Answering on the ScienceQA dataset using a two-leg CNN + text encoder architecture with three interchangeable fusion strategies.

## Architecture

```
Image ──► ResNet-50 (layer4 → [B,2048,7,7]) ──► proj [B, D]
                                                       │
                                               ┌───────┴────────┐
                                               │  Fusion module  │
                                               └───────┬────────┘
Text (Q + choice_i) ──► DistilBERT ──► CLS [B*K, D] ──┘
                                               │
                                        logits [B, K]  →  argmax  →  choice index
```

Each question is scored as K independent (question, choice_i) pairs, so the model handles variable numbers of choices without a fixed output head. ~51% of samples have no image; a `has_image` gate zeroes out the visual contribution for text-only questions in all three fusion variants.

| `--fusion`    | Mechanism |
|--------------|-----------|
| `early`       | concat([vis, text]) → MLP → scalar per choice |
| `late`        | separate MLP per modality, combined via learned sigmoid gate |
| `cross_modal` | multi-head attention: query = text CLS, key/value = ResNet spatial tokens |

The cross-modal variant exposes per-region attention weights [B×K, heads, 7, 7] that are used for visualisation in `solution.ipynb`.

## Project structure

```
├── config.py
├── download_data.py
├── inspect_parquet.py
├── train.py
├── solution.ipynb          visualisations + submission CSV
├── requirements.txt
├── data/
│   ├── dataset.py
│   └── raw/                parquet files (gitignored)
├── models/
│   ├── vision_encoder.py
│   ├── text_encoder.py
│   ├── fusion.py
│   └── vqa_model.py
├── visualize/
│   ├── gradcam.py
│   └── feature_space.py
├── checkpoints/            (gitignored)
└── outputs/                (gitignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data

**Option A — Kaggle download**

Place `~/.kaggle/kaggle.json`, then:

```bash
python download_data.py
```

**Option B — manual**

Copy the three files into `data/raw/`:
```
scienceQA_train.parquet
scienceQA_val.parquet
scienceQA_test_set.parquet
```

Verify the schema once before training:
```bash
python inspect_parquet.py
```

## Training

```bash
python train.py --fusion early
python train.py --fusion late
python train.py --fusion cross_modal
```

## Results and submission

Open `solution.ipynb` and run all cells.  It loads all three trained models, produces GradCAM and feature space visualisations, and writes `outputs/submission.csv` from the best-performing variant.