"""
visualize/feature_space.py

Functions for comparing visual and textual feature spaces.
Called from solution.ipynb — not intended as a standalone script.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from tqdm import tqdm


@torch.no_grad()
def extract_embeddings(model, loader, device, max_samples=2000):
    """
    Extract paired (visual, text) embeddings from the validation set.
    Text embedding = encoding of the correct answer choice.

    Returns:
        visual_embs : [N, D]
        text_embs   : [N, D]
        subjects    : list[str]
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

        v_feat = model.vision_encoder(image)

        ans_ids  = input_ids[torch.arange(B), answer].to(device)
        ans_mask = attention_mask[torch.arange(B), answer].to(device)
        t_feat   = model.text_encoder(ans_ids, ans_mask)

        visual_list.append(v_feat.cpu().float().numpy())
        text_list.append(t_feat.cpu().float().numpy())
        subject_list.extend(batch["subject"])
        count += B

    return (
        np.concatenate(visual_list, axis=0),
        np.concatenate(text_list,   axis=0),
        subject_list,
    )


def cosine_analysis(visual_embs, text_embs, n_samples=1000, save_path=None):
    """
    Paired vs. unpaired cosine similarity distributions.
    Positive delta = the two modality spaces are partially aligned.
    """
    idx = np.random.choice(len(visual_embs),
                           min(n_samples, len(visual_embs)), replace=False)
    v, t = visual_embs[idx], text_embs[idx]

    paired_sims = np.array([
        sk_cosine(v[i:i+1], t[i:i+1])[0, 0]
        for i in range(len(idx))
    ], dtype=float)

    shuf = np.random.permutation(len(idx))

    unpaired_sims = np.array([
        sk_cosine(v[i:i+1], t[shuf[i]:shuf[i]+1])[0, 0]
        for i in range(len(idx))
    ], dtype=float)

    stats = {
        "paired_mean":   float(np.mean(paired_sims)),
        "paired_std":    float(np.std(paired_sims)),
        "unpaired_mean": float(np.mean(unpaired_sims)),
        "unpaired_std":  float(np.std(unpaired_sims)),
        "delta":         float(np.mean(paired_sims) - np.mean(unpaired_sims)),
    }
    print(f"Paired   mean={stats['paired_mean']:.4f}  std={stats['paired_std']:.4f}")
    print(f"Unpaired mean={stats['unpaired_mean']:.4f}  std={stats['unpaired_std']:.4f}")
    print(f"Delta (paired - unpaired): {stats['delta']:+.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-0.3, 1.0, 60)
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
    plt.show()
    plt.close()
    return stats


def umap_joint(visual_embs, text_embs, subjects, n_max=1500, save_path=None):
    """
    Joint 2-D UMAP of visual and text embeddings.
    Circle = visual, triangle = text. Colour = ScienceQA subject.
    """
    try:
        from umap import UMAP
    except ImportError:
        print("umap-learn not installed: pip install umap-learn --break-system-packages")
        return

    n   = min(n_max, len(visual_embs))
    idx = np.random.choice(len(visual_embs), n, replace=False)
    v, t, subs = visual_embs[idx], text_embs[idx], [subjects[i] for i in idx]

    print("Running UMAP...")
    emb  = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                random_state=42).fit_transform(np.concatenate([v, t], axis=0))
    v2, t2 = emb[:n], emb[n:]

    palette = {
        "natural science":  "#378ADD",
        "social science":   "#D85A30",
        "language science": "#1D9E75",
        "unknown":          "#888780",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(v2[:, 0], v2[:, 1], s=8, alpha=0.5, color="#378ADD", label="visual")
    axes[0].scatter(t2[:, 0], t2[:, 1], s=8, alpha=0.5, color="#D85A30", label="text")
    axes[0].set_title("UMAP — by modality")
    axes[0].legend(markerscale=2, fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    for subj in sorted(set(subs)):
        mask = np.array([s == subj for s in subs])
        c    = palette.get(subj, "#888780")
        axes[1].scatter(v2[mask, 0], v2[mask, 1], s=8, alpha=0.5,
                        color=c, marker="o", label=f"{subj} (V)")
        axes[1].scatter(t2[mask, 0], t2[mask, 1], s=8, alpha=0.5,
                        color=c, marker="^")
    axes[1].set_title("UMAP — by subject  (● visual  ▲ text)")
    axes[1].legend(fontsize=7, markerscale=2)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
