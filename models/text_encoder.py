import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    DistilBERT text encoder (transformer, permitted — the non-transformer
    constraint applies only to the vision backbone).

    Encodes a batch of (question + choice) strings and returns:
      - CLS token projected to fusion_dim  [B, fusion_dim]
      - (optionally) all token hidden states [B, seq_len, hidden]
    """

    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.text_model)
        hidden = self.encoder.config.hidden_size      # 768 for distilbert-base

        self.proj    = nn.Sequential(
            nn.Linear(hidden, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids:      torch.Tensor,   # [N, seq_len]
        attention_mask: torch.Tensor,   # [N, seq_len]
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT: last_hidden_state shape [N, seq_len, hidden]
        hidden = out.last_hidden_state

        cls = hidden[:, 0, :]                          # [N, hidden]
        projected = self.proj(self.dropout(cls))       # [N, fusion_dim]

        if return_all_tokens:
            return projected, hidden
        return projected
