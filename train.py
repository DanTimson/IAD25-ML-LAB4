"""
train.py — train a VQA model for one fusion variant.

Usage:
    python train.py --fusion early
    python train.py --fusion late
    python train.py --fusion cross_modal
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from config import Config
from data.dataset import ScienceQADataset
from models.vqa_model import VQAModel


# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: VQAModel, config: Config) -> AdamW:
    """Separate learning rates for pretrained backbones vs. new fusion layers."""
    backbone_params = (
        list(model.vision_encoder.parameters())
        + list(model.text_encoder.parameters())
    )
    fusion_params = list(model.fusion.parameters())
    return AdamW(
        [
            {"params": backbone_params, "lr": config.backbone_lr},
            {"params": fusion_params,   "lr": config.fusion_lr},
        ],
        weight_decay=config.weight_decay,
    )


# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, device, scaler, config):
    model.train()
    total_loss = 0.0
    n_correct  = 0
    n_total    = 0
    criterion  = nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc="  train", leave=False):
        image          = batch["image"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        has_image      = batch["has_image"].to(device)
        choice_mask    = batch["choice_mask"].to(device)
        answer         = batch["answer"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            logits = model(image, input_ids, attention_mask, has_image, choice_mask)
            loss   = criterion(logits, answer)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n_correct  += (logits.argmax(-1) == answer).sum().item()
        n_total    += answer.size(0)

    return total_loss / len(loader), n_correct / n_total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n_correct = 0
    n_total   = 0

    for batch in tqdm(loader, desc="  eval ", leave=False):
        image          = batch["image"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        has_image      = batch["has_image"].to(device)
        choice_mask    = batch["choice_mask"].to(device)
        answer         = batch["answer"].to(device)

        logits    = model(image, input_ids, attention_mask, has_image, choice_mask)
        n_correct += (logits.argmax(-1) == answer).sum().item()
        n_total   += answer.size(0)

    return n_correct / n_total


# ──────────────────────────────────────────────────────────────────────────────
def main(fusion_type: str):
    config = Config(fusion_type=fusion_type)
    set_seed(config.seed)

    os.makedirs(config.output_dir,     exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  fusion={fusion_type}")

    tokenizer  = AutoTokenizer.from_pretrained(config.text_model)
    train_set  = ScienceQADataset("train",      config, tokenizer)
    val_set    = ScienceQADataset("validation", config, tokenizer)

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    model     = VQAModel(config).to(device)
    optimizer = build_optimizer(model, config)
    scaler    = torch.amp.GradScaler("cuda")

    total_steps = len(train_loader) * config.num_epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    history      = []
    best_val_acc = 0.0

    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, config
        )
        val_acc = evaluate(model, val_loader, device)

        row = {
            "epoch":     epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc,  4),
            "val_acc":    round(val_acc,     4),
        }
        history.append(row)
        print(f"  loss={train_loss:.4f}  train={train_acc:.4f}  val={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(config.checkpoint_dir, f"best_{fusion_type}.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "config":      config.__dict__,
            }, ckpt_path)
            print(f"  ✓ saved checkpoint → {ckpt_path}")

    hist_path = os.path.join(config.output_dir, f"history_{fusion_type}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val accuracy [{fusion_type}]: {best_val_acc:.4f}")
    return best_val_acc


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fusion",
        choices=["early", "late", "cross_modal"],
        default="early",
        help="Fusion strategy to train",
    )
    args = parser.parse_args()
    main(args.fusion)
