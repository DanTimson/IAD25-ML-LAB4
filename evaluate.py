"""
evaluate.py — generate test-set submission CSV and compare fusion variants.

Usage:
    # Generate submission for one variant
    python evaluate.py --fusion cross_modal

    # Compare all three variants (val set)
    python evaluate.py --compare
"""

import argparse
import csv
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from config import Config
from data.dataset import ScienceQADataset
from models.vqa_model import VQAModel


# ──────────────────────────────────────────────────────────────────────────────
def load_model(fusion_type: str, device: torch.device) -> tuple[VQAModel, Config]:
    config = Config(fusion_type=fusion_type)
    model  = VQAModel(config).to(device)
    ckpt   = torch.load(
        os.path.join(config.checkpoint_dir, f"best_{fusion_type}.pt"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config


@torch.no_grad()
def generate_submission(fusion_type: str):
    """Write outputs/submission_{fusion_type}.csv for the test split."""
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(fusion_type, device)
    tokenizer = AutoTokenizer.from_pretrained(config.text_model)
    test_set  = ScienceQADataset("test", config, tokenizer)
    loader    = DataLoader(test_set, batch_size=config.batch_size * 2,
                           shuffle=False, num_workers=4)

    rows = []
    for batch in tqdm(loader, desc=f"Inference [{fusion_type}]"):
        image          = batch["image"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        has_image      = batch["has_image"].to(device)
        choice_mask    = batch["choice_mask"].to(device)

        logits = model(image, input_ids, attention_mask, has_image, choice_mask)
        preds  = logits.argmax(-1).cpu().tolist()
        ids    = batch["task_id"]

        for tid, pred in zip(ids, preds):
            rows.append({"ID": int(tid), "answer": pred})

    rows.sort(key=lambda r: r["ID"])

    os.makedirs("outputs", exist_ok=True)
    out = f"outputs/submission_{fusion_type}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "answer"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved → {out}  ({len(rows)} rows)")
    return out


# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compare_variants():
    """
    Evaluate all three fusion variants on the validation split and print a
    comparison table: accuracy, inference time, parameter count.
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants  = ["early", "late", "cross_modal"]
    results   = {}

    for ftype in variants:
        ckpt_path = f"checkpoints/best_{ftype}.pt"
        if not os.path.exists(ckpt_path):
            print(f"[skip] no checkpoint for {ftype}")
            continue

        model, config = load_model(ftype, device)
        tokenizer     = AutoTokenizer.from_pretrained(config.text_model)
        val_set       = ScienceQADataset("validation", config, tokenizer)
        loader        = DataLoader(val_set, batch_size=config.batch_size * 2,
                                   shuffle=False, num_workers=4)

        n_params   = sum(p.numel() for p in model.parameters()) / 1e6
        n_correct  = 0
        n_total    = 0
        t_start    = time.perf_counter()

        for batch in tqdm(loader, desc=f"  [{ftype}]", leave=False):
            image          = batch["image"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            has_image      = batch["has_image"].to(device)
            choice_mask    = batch["choice_mask"].to(device)
            answer         = batch["answer"].to(device)

            logits    = model(image, input_ids, attention_mask, has_image, choice_mask)
            n_correct += (logits.argmax(-1) == answer).sum().item()
            n_total   += answer.size(0)

        elapsed   = time.perf_counter() - t_start
        val_acc   = n_correct / n_total
        ms_per_ex = elapsed / n_total * 1000

        # ── per-subject breakdown ─────────────────────────────────────────
        subject_acc = _subject_breakdown(model, config, tokenizer, device)

        results[ftype] = {
            "val_acc":       round(val_acc,   4),
            "params_M":      round(n_params,  2),
            "ms_per_sample": round(ms_per_ex, 2),
            "subject_acc":   subject_acc,
        }

    # Print table
    print("\n" + "=" * 65)
    print(f"{'Fusion':<14} {'Val Acc':>8} {'Params(M)':>10} {'ms/sample':>10}")
    print("-" * 65)
    for ftype, r in results.items():
        print(f"{ftype:<14} {r['val_acc']:>8.4f} {r['params_M']:>10.1f} {r['ms_per_sample']:>10.2f}")
    print("=" * 65)

    print("\nPer-subject accuracy:")
    subjects = ["natural science", "social science", "language science"]
    print(f"{'Fusion':<14} " + "  ".join(f"{s[:12]:>12}" for s in subjects))
    print("-" * 65)
    for ftype, r in results.items():
        row = "  ".join(f"{r['subject_acc'].get(s, 0.0):>12.4f}" for s in subjects)
        print(f"{ftype:<14} {row}")

    out = "outputs/comparison.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


@torch.no_grad()
def _subject_breakdown(model, config, tokenizer, device) -> dict:
    """Accuracy split by ScienceQA subject."""
    from datasets import load_dataset
    val_data = load_dataset(config.dataset_name, split="validation",
                            trust_remote_code=True)

    val_set = ScienceQADataset("validation", config, tokenizer)
    loader  = DataLoader(val_set, batch_size=config.batch_size * 2,
                         shuffle=False, num_workers=4)

    correct_by_sub = {}
    total_by_sub   = {}
    idx            = 0

    for batch in loader:
        image          = batch["image"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        has_image      = batch["has_image"].to(device)
        choice_mask    = batch["choice_mask"].to(device)
        answer         = batch["answer"]

        logits = model(image, input_ids, attention_mask, has_image, choice_mask)
        preds  = logits.argmax(-1).cpu()

        for i in range(len(preds)):
            subject = val_data[idx]["subject"]
            correct_by_sub[subject] = correct_by_sub.get(subject, 0) + int(preds[i] == answer[i])
            total_by_sub[subject]   = total_by_sub.get(subject, 0) + 1
            idx += 1

    return {s: correct_by_sub[s] / total_by_sub[s] for s in total_by_sub}


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion",   choices=["early", "late", "cross_modal"],
                        default=None)
    parser.add_argument("--compare",  action="store_true",
                        help="Compare all three variants on the validation set")
    args = parser.parse_args()

    if args.compare:
        compare_variants()
    elif args.fusion:
        generate_submission(args.fusion)
    else:
        parser.print_help()
