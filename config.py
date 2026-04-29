import os
from dataclasses import dataclass


@dataclass
class Config:
    # ---------- data paths ----------
    # Either place files manually under data/raw/ or run download_data.py.
    data_dir:         str = "data/raw"

    @property
    def train_parquet(self) -> str:
        return os.path.join(self.data_dir, "scienceQA_train.parquet")

    @property
    def val_parquet(self) -> str:
        return os.path.join(self.data_dir, "scienceQA_val.parquet")

    @property
    def test_parquet(self) -> str:
        return os.path.join(self.data_dir, "scienceQA_test_set.parquet")

    # ---------- preprocessing ----------
    max_seq_len:  int = 128
    image_size:   int = 224
    max_choices:  int = 5       # ScienceQA has 2–5 options; pad to this

    # Context fields to include in the text encoding.
    # hint    — short contextual note, often empty; low risk, recommended on.
    # lecture — long background text (300+ tokens); truncated to max_seq_len,
    #           but competes with question+choice for the token budget.
    #           Run an ablation before enabling in final submission.
    use_hint:    bool = True
    use_lecture: bool = False

    # ---------- model ----------
    vision_backbone: str = "resnet50"
    text_model:      str = "distilbert-base-uncased"
    vision_dim:      int = 2048      # ResNet-50 layer4 channels
    text_dim:        int = 768       # DistilBERT hidden size
    fusion_dim:      int = 512
    n_attn_heads:    int = 8
    dropout:         float = 0.3

    # ---------- training ----------
    batch_size:    int   = 16
    num_epochs:    int   = 10
    backbone_lr:   float = 5e-5
    fusion_lr:     float = 2e-4
    weight_decay:  float = 1e-2
    warmup_steps:  int   = 200
    grad_clip:     float = 1.0
    seed:          int   = 42

    # ---------- fusion variant ----------
    # one of: "early" | "late" | "cross_modal"
    fusion_type: str = "early"

    # ---------- output paths ----------
    output_dir:     str = "outputs"
    checkpoint_dir: str = "checkpoints"