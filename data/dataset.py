import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image
import numpy as np


# ImageNet normalisation stats
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
        T.Resize(int(image_size * 1.1)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])


class ScienceQADataset(Dataset):
    """
    Wraps the HuggingFace ScienceQA dataset.

    Each item returns:
      image         : [3, H, W] float tensor  (zeros if no image context)
      has_image     : scalar float 0/1 — used by fusion modules to gate visual contribution
      input_ids     : [max_choices, seq_len]  — one encoding per choice
      attention_mask: [max_choices, seq_len]
      choice_mask   : [max_choices]  — 1 for real choices, 0 for padding
      answer        : scalar long     — correct choice index
      task_id       : int             — original dataset id for submission file
    """

    def __init__(self, split: str, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.transform = build_transform(config.image_size, train=(split == "train"))
        self.null_image = torch.zeros(3, config.image_size, config.image_size)

        # HuggingFace split names: "train" | "validation" | "test"
        self.data = load_dataset(config.dataset_name, split=split,
                                 trust_remote_code=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]

        # ---- image -------------------------------------------------------
        raw_image = sample.get("image")
        if raw_image is not None:
            if not isinstance(raw_image, Image.Image):
                raw_image = Image.fromarray(np.array(raw_image))
            image = self.transform(raw_image.convert("RGB"))
            has_image = torch.tensor(1.0)
        else:
            image = self.null_image.clone()
            has_image = torch.tensor(0.0)

        # ---- text --------------------------------------------------------
        question = sample["question"]
        choices  = sample["choices"]          # list[str], length 2–5
        n_valid  = len(choices)

        # Format: the model scores each (question, choice) pair independently.
        # Padding choices are empty strings; they are masked out during scoring.
        texts = [f"Question: {question} Choice: {c}" for c in choices]
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
        choice_mask[:n_valid] = 1.0

        # task_id: ScienceQA uses field "pid" (problem id); fall back to index
        task_id = sample.get("pid", idx)
        if isinstance(task_id, str):
            task_id = int(task_id)

        return {
            "image":          image,                        # [3, H, W]
            "has_image":      has_image,                    # scalar
            "input_ids":      enc["input_ids"],             # [max_choices, seq_len]
            "attention_mask": enc["attention_mask"],        # [max_choices, seq_len]
            "choice_mask":    choice_mask,                  # [max_choices]
            "answer":         torch.tensor(sample["answer"], dtype=torch.long),
            "task_id":        task_id,
        }