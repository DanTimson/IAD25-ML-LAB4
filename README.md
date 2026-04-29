# ScienceQA VQA

Visual Question Answering on the [ScienceQA](https://scienceqa.github.io/) dataset using a two-leg CNN + text encoder architecture with three interchangeable fusion strategies.

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
├── download_data.py        download competition files from Kaggle
├── inspect_parquet.py      schema verification (run once after download)
├── train.py                train one fusion variant
├── solution.ipynb          post-training: visualisations + submission CSV
├── requirements.txt
├── data/
│   ├── dataset.py
│   └── raw/                parquet files (gitignored)
├── models/
│   ├── vision_encoder.py   ResNet-50 with GradCAM hooks
│   ├── text_encoder.py     DistilBERT
│   ├── fusion.py           EarlyFusion, LateFusion, CrossModalFusion
│   └── vqa_model.py
├── visualize/
│   ├── gradcam.py          GradCAM + question-conditioned attention
│   └── feature_space.py    cosine similarity + UMAP
├── checkpoints/            saved by train.py (gitignored)
└── outputs/                plots, JSON, submission CSV (gitignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data

**Option A — Kaggle download**

Place `~/.kaggle/kaggle.json` (from https://www.kaggle.com/settings → API → Create New Token), then:

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

Open `solution.ipynb` and run all cells. It will:

1. Load all three trained models
2. Report validation accuracy per variant and per subject
3. Produce GradCAM visualisations (saved to `outputs/`)
4. Show question-conditioned attention shift for the cross-modal model
5. Run cosine similarity and UMAP analysis of the feature spaces
6. Select the best-performing variant and write `outputs/submission.csv`