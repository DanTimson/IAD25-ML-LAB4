import os
from dataclasses import dataclass


@dataclass
class Config:
    data_dir: str = "data/raw"

    @property
    def train_parquet(self) -> str:
        return os.path.join(self.data_dir, "scienceQA_train.parquet")

    @property
    def val_parquet(self) -> str:
        return os.path.join(self.data_dir, "scienceQA_val.parquet")

    @property
    def test_parquet(self) -> str:
        return os.path.join(self.data_dir, "scienceQA_test_set.parquet")

    max_seq_len: int = 128
    image_size:  int = 224
    max_choices: int = 5

    # hint is short and low-risk; lecture is long and competes with the
    # question+choice for the token budget — disabled by default
    use_hint:    bool = True
    use_lecture: bool = False

    text_model:   str = "distilbert-base-uncased"
    fusion_dim:   int = 512
    n_attn_heads: int = 8
    dropout:      float = 0.3

    batch_size:   int   = 16
    num_epochs:   int   = 10
    backbone_lr:  float = 5e-5
    fusion_lr:    float = 2e-4
    weight_decay: float = 1e-2
    warmup_steps: int   = 200
    grad_clip:    float = 1.0
    seed:         int   = 42

    fusion_type: str = "early"

    output_dir:     str = "outputs"
    checkpoint_dir: str = "checkpoints"