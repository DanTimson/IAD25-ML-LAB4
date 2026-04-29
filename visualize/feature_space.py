"""
visualize/feature_space.py

Answers rubric criterion 4: "compare result feature space for visual and textual
parts — how similar are they?"

Two analyses:
  1. Cosine similarity distribution — paired (matched image–text) vs. unpaired (random).
  2. Joint UMAP of visual and text embeddings coloured by ScienceQA subject.

Usage (from project root):
    python -c "
    from visualize.feature_space import run_analysis
    run_analysis('cross_modal')
    "
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine


# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(
    model,
    loader: DataLoader,
    device,
    max_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract matched (visual, text) embeddings for validation samples.

    For each sample we use the encoding of the *correct* answer choice as the
    text embedding so that paired similarity has a meaningful interpretation.

    Returns:
        visual_embs : [N, D]
        text_embs   : [N, D]
        subjects    : list[str]  (ScienceQA subject label per sample)
    """
    model.eval()
    visual_list, text_list, subject_list = [], [], []
    count = 0

    for batch in tqdm(loader, desc="Extracting embeddings"):
        if count >= max_samples:
            break

        image          = batch["image"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answer         = batch["answer"]
        B, K, L        = input_ids.shape

        # Visual features (global pooled projection)
        v_feat = model.vision_encoder(image)   # [B, D]

        # Text: encode only the correct choice
        ans_ids  = input_ids[torch.arange(B), answer].to(device)   # [B, L]
        ans_mask = attention_mask[torch.arange(B), answer].to(device)
        t_feat   = model.text_encoder(ans_ids, ans_mask)            # [B, D]

        visual_list.append(v_feat.cpu().float().numpy())
        text_list.append(t_feat.cpu().float().numpy())

        if "subject" in batch:
            subject_list.extend(batch["subject"])
        else:
            subject_list.extend(["unknown"] * B)

        count += B

    return (np.concatenate(visual_list, axis=0),
            np.concatenate(text_list,   axis=0),
            subject_list)


# ──────────────────────────────────────────────────────────────────────────────
def cosine_analysis(
    visual_embs: np.ndarray,
    text_embs:   np.ndarray,
    n_samples:   int = 1000,
    save_path:   str | None = None,
):
    """
    Paired vs. unpaired cosine similarity distributions.
    Quantifies whether the visual and text spaces are aligned.
    """
    idx = np.random.choice(len(visual_embs), min(n_samples, len(visual_embs)), replace=False)
    v   = visual_embs[idx]
    t   = text_embs[idx]

    # Paired: each image with its own question+answer text
    paired_sims = np.array([
        float(sk_cosine(v[i:i+1], t[i:i+1]))
        for i in range(len(idx))
    ])

    # Unpaired: each image with a randomly chosen text
    shuf = np.random.permutation(len(idx))
    unpaired_sims = np.array([
        float(sk_cosine(v[i:i+1], t[shuf[i]:shuf[i]+1]))
        for i in range(len(idx))
    ])

    stats = {
        "paired_mean":   float(np.mean(paired_sims)),
        "paired_std":    float(np.std(paired_sims)),
        "unpaired_mean": float(np.mean(unpaired_sims)),
        "unpaired_std":  float(np.std(unpaired_sims)),
        "delta_mean":    float(np.mean(paired_sims) - np.mean(unpaired_sims)),
    }
    print("\n── Cosine similarity ──────────────────────────────")
    print(f"  Paired   : mean={stats['paired_mean']:.4f}  std={stats['paired_std']:.4f}")
    print(f"  Unpaired : mean={stats['unpaired_mean']:.4f}  std={stats['unpaired_std']:.4f}")
    print(f"  Δ (paired − unpaired): {stats['delta_mean']:+.4f}")
    print("  Positive Δ → visual and text spaces are partially aligned.")

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-0.2, 1.0, 60)
    ax.hist(paired_sims,   bins=bins, alpha=0.65, label="paired",   color="#378ADD")
    ax.hist(unpaired_sims, bins=bins, alpha=0.65, label="unpaired", color="#D85A30")
    ax.axvline(stats["paired_mean"],   color="#185FA5", lw=1.5, linestyle="--")
    ax.axvline(stats["unpaired_mean"], color="#993C1D", lw=1.5, linestyle="--")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Visual–text cosine similarity: paired vs. unpaired")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    plt.close()
    return stats


# ──────────────────────────────────────────────────────────────────────────────
def umap_joint(
    visual_embs: np.ndarray,
    text_embs:   np.ndarray,
    subjects:    list[str],
    n_max:       int = 1500,
    save_path:   str | None = None,
):
    """
    Joint 2-D UMAP of visual and text embeddings.
    Colour by ScienceQA subject; marker shape by modality.
    """
    try:
        from umap import UMAP
    except ImportError:
        print("[skip] umap-learn not installed. Run: pip install umap-learn --break-system-packages")
        return

    n = min(n_max, len(visual_embs))
    idx     = np.random.choice(len(visual_embs), n, replace=False)
    v_sub   = visual_embs[idx]
    t_sub   = text_embs[idx]
    sub_sub = [subjects[i] for i in idx]

    joint = np.concatenate([v_sub, t_sub], axis=0)   # [2n, D]
    print("Running UMAP (this may take 30–90 s) …")
    emb   = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                 random_state=42).fit_transform(joint)

    v_emb = emb[:n]
    t_emb = emb[n:]

    unique_subjects = sorted(set(sub_sub))
    palette = {"natural science": "#378ADD",
               "social science":  "#D85A30",
               "language science":"#1D9E75",
               "unknown":         "#888780"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: colour by modality
    axes[0].scatter(v_emb[:, 0], v_emb[:, 1], s=8, alpha=0.5,
                    label="visual", color="#378ADD")
    axes[0].scatter(t_emb[:, 0], t_emb[:, 1], s=8, alpha=0.5,
                    label="text",   color="#D85A30")
    axes[0].set_title("UMAP — by modality")
    axes[0].legend(markerscale=2, fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Right: colour by subject
    for subj in unique_subjects:
        mask = np.array([s == subj for s in sub_sub])
        c    = palette.get(subj, "#888780")
        axes[1].scatter(v_emb[mask, 0], v_emb[mask, 1], s=8, alpha=0.5,
                        color=c, marker="o", label=f"{subj} (V)")
        axes[1].scatter(t_emb[mask, 0], t_emb[mask, 1], s=8, alpha=0.5,
                        color=c, marker="^")
    axes[1].set_title("UMAP — by subject (● visual, ▲ text)")
    axes[1].legend(fontsize=7, markerscale=2)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
def run_analysis(fusion_type: str = "cross_modal", max_samples: int = 2000):
    """End-to-end: load model → extract embeddings → run both analyses."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from config import Config
    from data.dataset import ScienceQADataset
    from models.vqa_model import VQAModel

    config    = Config(fusion_type=fusion_type)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(config.text_model)

    val_set = ScienceQADataset("validation", config, tokenizer)
    loader  = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    model = VQAModel(config).to(device)
    ckpt  = torch.load(f"checkpoints/best_{fusion_type}.pt",
                       map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    visual_embs, text_embs, subjects = extract_embeddings(
        model, loader, device, max_samples=max_samples
    )

    os.makedirs("outputs", exist_ok=True)
    stats = cosine_analysis(
        visual_embs, text_embs,
        save_path=f"outputs/cosine_{fusion_type}.png",
    )
    umap_joint(
        visual_embs, text_embs, subjects,
        save_path=f"outputs/umap_{fusion_type}.png",
    )

    with open(f"outputs/feature_stats_{fusion_type}.json", "w") as f:
        json.dump(stats, f, indent=2)
