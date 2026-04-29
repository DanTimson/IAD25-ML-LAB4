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
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(_STD * img + _MEAN, 0, 1)


def _overlay(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = img.shape[:2]
    hm = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8))
             .resize((w, h), Image.Resampling.BILINEAR)
    ) / 255.0
    return np.clip(alpha * cm.jet(hm)[:, :, :3] + (1 - alpha) * img, 0, 1)


def gradcam_heatmap(model, image, input_ids, attn_mask, has_image, choice_mask,
                    target_choice=None, device="cpu"):
    model.eval()
    image, input_ids, attn_mask, has_image, choice_mask = (
        t.to(device) for t in (image, input_ids, attn_mask, has_image, choice_mask)
    )

    logits = model(image, input_ids, attn_mask, has_image, choice_mask)
    if target_choice is None:
        target_choice = int(logits.argmax(-1).item())

    model.zero_grad()
    logits[0, target_choice].backward()

    acts  = model.vision_encoder.activations[0]
    grads = model.vision_encoder.gradients[0]

    cam = F.relu((grads.mean(dim=(1, 2), keepdim=True) * acts).sum(dim=0))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def cross_modal_heatmap(model, image, input_ids, attn_mask, has_image, choice_mask,
                        choice_idx=0, device="cpu"):
    assert model.fusion_type == "cross_modal"
    model.eval()
    with torch.no_grad():
        model(image.to(device), input_ids.to(device), attn_mask.to(device),
              has_image.to(device), choice_mask.to(device))

    attn_map = model.fusion.attention_weights[choice_idx].mean(dim=0).squeeze(0)
    heatmap  = attn_map.cpu().numpy().reshape(7, 7)
    heatmap  = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def plot_gradcam_grid(model, samples, device, save_path=None):
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.05})
    if n == 1:
        axes = axes[None]

    for i, s in enumerate(samples):
        img_t  = s["image"].unsqueeze(0)
        ids    = s["input_ids"].unsqueeze(0)
        mask   = s["attention_mask"].unsqueeze(0)
        has_im = s["has_image"].unsqueeze(0)
        cmask  = s["choice_mask"].unsqueeze(0)
        answer = s["answer"].item()

        cam     = gradcam_heatmap(model, img_t, ids, mask, has_im, cmask,
                                  target_choice=answer, device=device)
        orig    = _denormalize(s["image"])
        overlay = _overlay(orig, cam)

        for ax, title, img in zip(axes[i],
                                  ["original", "GradCAM", "overlay"],
                                  [orig, cm.jet(cam)[:, :, :3], overlay]):
            ax.imshow(img)
            ax.set_title(f"{title}  (ans={answer})", fontsize=9)
            ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_question_conditioned(model, image_tensor, tokenized_items, device, save_path=None):
    """Same image, different questions — shows whether attention is question-driven."""
    n = len(tokenized_items)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n),
                             gridspec_kw={"hspace": 0.4, "wspace": 0.05})
    if n == 1:
        axes = axes[None]

    orig = _denormalize(image_tensor)

    for i, item in enumerate(tokenized_items):
        img_t  = item["image"].unsqueeze(0)
        ids    = item["input_ids"].unsqueeze(0)
        mask   = item["attention_mask"].unsqueeze(0)
        has_im = item["has_image"].unsqueeze(0)
        cmask  = item["choice_mask"].unsqueeze(0)
        answer = item["answer"].item()

        cam     = gradcam_heatmap(model, img_t, ids, mask, has_im, cmask,
                                  target_choice=answer, device=device)
        overlay = _overlay(orig, cam)

        if model.fusion_type == "cross_modal":
            third = _overlay(orig, cross_modal_heatmap(
                model, img_t, ids, mask, has_im, cmask,
                choice_idx=answer, device=device,
            ))
            third_title = "cross-modal attn"
        else:
            third = cm.jet(cam)[:, :, :3]
            third_title = "GradCAM raw"

        for ax, title, panel in zip(axes[i],
                                    ["original", "GradCAM overlay", third_title],
                                    [orig, overlay, third]):
            ax.imshow(panel)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    plt.suptitle("Question-conditioned attention (same image)", fontsize=11, y=1.01)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()