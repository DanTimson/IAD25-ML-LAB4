import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.text_model)
        hidden = self.encoder.config.hidden_size

        self.proj = nn.Sequential(
            nn.Linear(hidden, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.proj(self.dropout(cls))