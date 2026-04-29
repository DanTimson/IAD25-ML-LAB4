"""
evaluate.py

Usage:
    # Generate submission CSV from test parquet
    python evaluate.py --fusion cross_modal

    # Compare all three variants on val set
    python evaluate.py --compare

    # Override data directory
    python evaluate.py --fusion early --data_dir /path/to/parquets
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


def load_model(fusion_type: str, device: torch.device,
               data_dir: str | None = None) -> tuple[VQAModel, Config]:
    config = Config(fusion_type=fusion_type)
    if data_dir:
        config.data_dir = data_dir
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
def generate_submission(fusion_type: str, data_dir: str | None = None):
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(fusion_type, device, data_dir)
    tokenizer     = AutoTokenizer.from_pretrained(config.text_model)

    test_set = ScienceQADataset(config.test_parquet, config, tokenizer, split="test")
    loader   = DataLoader(test_set, batch_size=config.batch_size * 2,
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

        for tid, pred in zip(batch["task_id"].tolist(), preds):
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


@torch.no_grad()
def compare_variants(data_dir: str | None = None):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = ["early", "late", "cross_modal"]
    results  = {}

    for ftype in variants:
        ckpt_path = f"checkpoints/best_{ftype}.pt"
        if not os.path.exists(ckpt_path):
            print(f"[skip] no checkpoint for {ftype}")
            continue

        model, config = load_model(ftype, device, data_dir)
        tokenizer     = AutoTokenizer.from_pretrained(config.text_model)
        val_set       = ScienceQADataset(config.val_parquet, config, tokenizer, split="val")
        loader        = DataLoader(val_set, batch_size=config.batch_size * 2,
                                   shuffle=False, num_workers=4)

        n_params         = sum(p.numel() for p in model.parameters()) / 1e6
        n_correct        = 0
        n_total          = 0
        correct_by_subj  = {}
        total_by_subj    = {}
        t_start          = time.perf_counter()

        for batch in tqdm(loader, desc=f"  [{ftype}]", leave=False):
            image          = batch["image"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            has_image      = batch["has_image"].to(device)
            choice_mask    = batch["choice_mask"].to(device)
            answer         = batch["answer"].to(device)

            logits  = model(image, input_ids, attention_mask, has_image, choice_mask)
            correct = (logits.argmax(-1) == answer).cpu()

            n_correct += correct.sum().item()
            n_total   += answer.size(0)

            for subj, hit in zip(batch["subject"], correct.tolist()):
                correct_by_subj[subj] = correct_by_subj.get(subj, 0) + int(hit)
                total_by_subj[subj]   = total_by_subj.get(subj, 0) + 1

        elapsed = time.perf_counter() - t_start
        results[ftype] = {
            "val_acc":       round(n_correct / n_total, 4),
            "params_M":      round(n_params, 2),
            "ms_per_sample": round(elapsed / n_total * 1000, 2),
            "subject_acc":   {s: round(correct_by_subj[s] / total_by_subj[s], 4)
                              for s in total_by_subj},
        }

    print("\n" + "=" * 60)
    print(f"{'Fusion':<14} {'Val Acc':>8} {'Params(M)':>10} {'ms/sample':>10}")
    print("-" * 60)
    for ftype, r in results.items():
        print(f"{ftype:<14} {r['val_acc']:>8.4f} {r['params_M']:>10.1f} {r['ms_per_sample']:>10.2f}")
    print("=" * 60)

    subjects = ["natural science", "social science", "language science"]
    print("\nPer-subject accuracy:")
    print(f"{'Fusion':<14} " + "  ".join(f"{s[:16]:>16}" for s in subjects))
    print("-" * 70)
    for ftype, r in results.items():
        row = "  ".join(f"{r['subject_acc'].get(s, float('nan')):>16.4f}" for s in subjects)
        print(f"{ftype:<14} {row}")

    os.makedirs("outputs", exist_ok=True)
    out = "outputs/comparison.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion",   choices=["early", "late", "cross_modal"],
                        default=None)
    parser.add_argument("--compare",  action="store_true")
    parser.add_argument("--data_dir", default=None,
                        help="Override data directory (default: data/raw)")
    args = parser.parse_args()

    if args.compare:
        compare_variants(args.data_dir)
    elif args.fusion:
        generate_submission(args.fusion, args.data_dir)
    else:
        parser.print_help()