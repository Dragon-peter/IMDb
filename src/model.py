from __future__ import annotations

import torch
from torch import nn


class BiLSTMSentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_index: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 4, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, (hidden, _) = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.size(1),
        )

        mask = (input_ids != self.embedding.padding_idx).unsqueeze(-1)
        masked_output = unpacked * mask
        token_counts = mask.sum(dim=1).clamp(min=1)
        mean_pool = masked_output.sum(dim=1) / token_counts

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)

        features = torch.cat([mean_pool, final_hidden], dim=1)
        logits = self.classifier(self.dropout(features)).squeeze(1)
        return logits
