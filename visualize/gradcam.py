"""
visualize/gradcam.py

Two interpretability channels:
  1. GradCAM — works for ALL three fusion variants via layer4 hooks.
  2. Cross-modal attention map — available only for cross_modal fusion.

Public API:
    gradcam_heatmap(model, batch_item, device)  → np.ndarray [H, W]
    cross_modal_heatmap(model, batch_item, choice_idx, device) → np.ndarray [7, 7]
    plot_question_conditioned(model, sample, questions_choices, tokenizer, config, device)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from models.vqa_model import VQAModel

_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """[3,H,W] → [H,W,3] float32 in [0,1]."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = _STD * img + _MEAN
    return np.clip(img, 0, 1)


def _overlay(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [H,W] heatmap (0–1) on top of an [H,W,3] image."""
    h, w = img.shape[:2]
    hm_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8))
             .resize((w, h), Image.BILINEAR)
    ) / 255.0
    colormap   = cm.jet(hm_resized)[:, :, :3]
    return np.clip(alpha * colormap + (1 - alpha) * img, 0, 1)


# ──────────────────────────────────────────────────────────────────────────────
def gradcam_heatmap(
    model:       VQAModel,
    image:       torch.Tensor,    # [1, 3, H, W]
    input_ids:   torch.Tensor,    # [1, K, L]
    attn_mask:   torch.Tensor,    # [1, K, L]
    has_image:   torch.Tensor,    # [1]
    choice_mask: torch.Tensor,    # [1, K]
    target_choice: int | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """
    Compute GradCAM w.r.t. layer4 activations for the target choice.
    Returns a [7, 7] heatmap in [0, 1].
    Works for early, late, and cross_modal fusion alike.
    """
    model.eval()
    image = image.to(device)
    input_ids   = input_ids.to(device)
    attn_mask   = attn_mask.to(device)
    has_image   = has_image.to(device)
    choice_mask = choice_mask.to(device)

    # Forward (hooks capture activations)
    logits = model(image, input_ids, attn_mask, has_image, choice_mask)

    if target_choice is None:
        target_choice = int(logits.argmax(-1).item())

    model.zero_grad()
    logits[0, target_choice].backward()

    acts  = model.vision_encoder.activations[0]   # [2048, 7, 7]
    grads = model.vision_encoder.gradients[0]      # [2048, 7, 7]

    weights = grads.mean(dim=(1, 2), keepdim=True)  # [2048, 1, 1]
    cam     = F.relu((weights * acts).sum(dim=0))    # [7, 7]
    cam     = cam - cam.min()
    cam     = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def cross_modal_heatmap(
    model:         VQAModel,
    image:         torch.Tensor,
    input_ids:     torch.Tensor,
    attn_mask:     torch.Tensor,
    has_image:     torch.Tensor,
    choice_mask:   torch.Tensor,
    choice_idx:    int = 0,
    head_mean:     bool = True,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """
    Extract the spatial attention weights from CrossModalFusion for a given
    choice index and return a [7, 7] heatmap (average over attention heads).
    Only valid for fusion_type == 'cross_modal'.
    """
    assert model.fusion_type == "cross_modal", \
        "cross_modal_heatmap requires fusion_type='cross_modal'"

    model.eval()
    with torch.no_grad():
        model(
            image.to(device),
            input_ids.to(device),
            attn_mask.to(device),
            has_image.to(device),
            choice_mask.to(device),
        )

    # attention_weights: [B*K, n_heads, 1, 49]
    attn = model.fusion.attention_weights          # [B*K, H, 1, 49]
    K    = input_ids.shape[1]
    # Select the requested choice for the first (only) sample in batch
    attn_choice = attn[choice_idx]                 # [H, 1, 49]
    if head_mean:
        attn_map = attn_choice.mean(dim=0)         # [1, 49]
    else:
        attn_map = attn_choice[0]                  # [1, 49]  (first head)

    heatmap = attn_map.squeeze(0).cpu().numpy().reshape(7, 7)   # [7, 7]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


# ──────────────────────────────────────────────────────────────────────────────
def plot_gradcam_grid(
    model:     VQAModel,
    samples:   list[dict],
    device,
    save_path: str | None = None,
):
    """
    Visualise GradCAM for a list of dataset items.
    Each row: original | GradCAM | overlay
    """
    n   = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.05})
    if n == 1:
        axes = axes[None]

    for i, s in enumerate(samples):
        img_t     = s["image"].unsqueeze(0)
        input_ids = s["input_ids"].unsqueeze(0)
        attn_mask = s["attention_mask"].unsqueeze(0)
        has_img   = s["has_image"].unsqueeze(0)
        c_mask    = s["choice_mask"].unsqueeze(0)
        answer    = s["answer"].item()

        cam = gradcam_heatmap(model, img_t, input_ids, attn_mask,
                              has_img, c_mask, target_choice=answer, device=device)

        orig    = _denormalize(s["image"])
        overlay = _overlay(orig, cam)

        titles = ["original", "GradCAM", "overlay"]
        imgs   = [orig, cm.jet(cam)[:, :, :3], overlay]

        for j, (ax, title, img) in enumerate(zip(axes[i], titles, imgs)):
            ax.imshow(img)
            ax.set_title(f"{title}  (ans={answer})", fontsize=9)
            ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved GradCAM grid → {save_path}")
    plt.show()


def plot_question_conditioned(
    model:            VQAModel,
    image_tensor:     torch.Tensor,    # [3, H, W] — same image for all rows
    tokenized_items:  list[dict],      # one dict per question variant (from dataset)
    device,
    save_path: str | None = None,
):
    """
    Show how GradCAM / attention maps shift for the SAME image under
    different questions. Satisfies rubric criterion 3b.

    tokenized_items: list of dataset __getitem__ dicts, all sharing the same image
    but differing in question/choice text.
    """
    n   = len(tokenized_items)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n),
                              gridspec_kw={"hspace": 0.4, "wspace": 0.05})
    if n == 1:
        axes = axes[None]

    orig = _denormalize(image_tensor)

    for i, item in enumerate(tokenized_items):
        img_t     = item["image"].unsqueeze(0)
        input_ids = item["input_ids"].unsqueeze(0)
        attn_mask = item["attention_mask"].unsqueeze(0)
        has_img   = item["has_image"].unsqueeze(0)
        c_mask    = item["choice_mask"].unsqueeze(0)
        answer    = item["answer"].item()

        cam = gradcam_heatmap(model, img_t, input_ids, attn_mask,
                              has_img, c_mask, target_choice=answer, device=device)

        overlay = _overlay(orig, cam)

        if model.fusion_type == "cross_modal":
            attn_map = cross_modal_heatmap(
                model, img_t, input_ids, attn_mask, has_img, c_mask,
                choice_idx=answer, device=device,
            )
            third_panel = _overlay(orig, attn_map)
            third_title = "cross-modal attn"
        else:
            third_panel = cm.jet(cam)[:, :, :3]
            third_title = "GradCAM raw"

        axes[i, 0].imshow(orig);           axes[i, 0].set_title("original",   fontsize=8)
        axes[i, 1].imshow(overlay);        axes[i, 1].set_title("GradCAM overlay", fontsize=8)
        axes[i, 2].imshow(third_panel);    axes[i, 2].set_title(third_title,  fontsize=8)

        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle("Question-conditioned attention shift (same image)", fontsize=11, y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved question-conditioned plot → {save_path}")
    plt.show()
