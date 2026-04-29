import ast
import io

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def build_transform(image_size: int, train: bool) -> T.Compose:
    if train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToTensor(),
            T.Normalize(_MEAN, _STD),
        ])
    return T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])


def _decode_image(raw) -> Image.Image | None:
    """Image column is dict{'bytes', 'path'} or None/NaN (text-only question)."""
    if raw is None or isinstance(raw, float):
        return None
    if isinstance(raw, dict):
        data = raw.get("bytes")
        if not data:
            return None
        try:
            return Image.open(io.BytesIO(data))
        except Exception:
            return None
    return None


def _parse_choices(raw) -> list[str]:
    if isinstance(raw, (list, np.ndarray)):
        return [str(c) for c in raw]
    if isinstance(raw, str):
        return [str(c) for c in ast.literal_eval(raw)]
    raise ValueError(f"Unexpected choices type {type(raw)}: {raw!r}")


def _build_text(question: str, choice: str, hint: str, lecture: str,
                use_hint: bool, use_lecture: bool) -> str:
    # Order: lecture -> hint -> question -> choice
    # Tokenizer truncates from the right, so question and choice are preserved.
    parts = []
    if use_lecture and lecture:
        parts.append(f"Context: {lecture}")
    if use_hint and hint:
        parts.append(f"Hint: {hint}")
    parts.append(f"Question: {question}")
    parts.append(f"Choice: {choice}")
    return " ".join(parts)


class ScienceQADataset(Dataset):
    """
    Parquet-backed dataset for all three splits.
    task_id is the 0-based row index for all splits (no dedicated ID column).
    """

    def __init__(self, parquet_path: str, config, tokenizer, split: str):
        assert split in ("train", "val", "test")
        self.config     = config
        self.tokenizer  = tokenizer
        self.transform  = build_transform(config.image_size, train=(split == "train"))
        self.null_image = torch.zeros(3, config.image_size, config.image_size)

        self.df          = pd.read_parquet(parquet_path)
        self._has_answer = "answer" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        pil = _decode_image(row["image"] if "image" in self.df.columns else None)
        if pil is not None:
            image     = self.transform(pil.convert("RGB"))
            has_image = torch.tensor(1.0)
        else:
            image     = self.null_image.clone()
            has_image = torch.tensor(0.0)

        question = str(row["question"])
        choices  = _parse_choices(row["choices"])
        hint     = str(row["hint"])    if row.get("hint")    else ""
        lecture  = str(row["lecture"]) if row.get("lecture") else ""

        texts = [
            _build_text(question, c, hint, lecture,
                        self.config.use_hint, self.config.use_lecture)
            for c in choices
        ]
        while len(texts) < self.config.max_choices:
            texts.append("")

        enc = self.tokenizer(
            texts,
            max_length=self.config.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        choice_mask = torch.zeros(self.config.max_choices)
        choice_mask[:len(choices)] = 1.0

        item = {
            "image":          image,
            "has_image":      has_image,
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "choice_mask":    choice_mask,
            "task_id":        idx,
            "subject":        str(row.get("subject", "unknown")),
        }
        if self._has_answer:
            item["answer"] = torch.tensor(int(row["answer"]), dtype=torch.long)
        return item