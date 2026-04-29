from dataclasses import dataclass, field


@dataclass
class Config:
    # ---------- data ----------
    dataset_name: str = "derek-thomas/ScienceQA"
    max_seq_len: int = 128
    image_size: int = 224
    max_choices: int = 5          # ScienceQA has 2–5 options; pad to this

    # ---------- model ----------
    vision_backbone: str = "resnet50"   # non-transformer; swap to efficientnet_b0 if desired
    text_model: str = "distilbert-base-uncased"
    vision_dim: int = 2048              # ResNet50 layer4 channels
    text_dim: int = 768                 # DistilBERT hidden size
    fusion_dim: int = 512               # shared projection dim
    n_attn_heads: int = 8               # cross-modal attention heads
    dropout: float = 0.3

    # ---------- training ----------
    batch_size: int = 16
    num_epochs: int = 10
    backbone_lr: float = 5e-5          # pretrained layers (vision + text encoders)
    fusion_lr: float = 2e-4            # new fusion layers
    weight_decay: float = 1e-2
    warmup_steps: int = 200
    grad_clip: float = 1.0
    seed: int = 42

    # ---------- fusion variant ----------
    # one of: "early" | "late" | "cross_modal"
    fusion_type: str = "early"

    # ---------- paths ----------
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
