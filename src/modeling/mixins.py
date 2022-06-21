from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/affjljoo3581/CommonLit-Readability-Prize/blob/c6b44a330e6cd37d9310e3ef0f530f9f429ffb05/src/modeling/miscellaneous.py#L62
class AttentionHead(nn.Module):
    """An attention-based classification head.
    This class is used to pool the hidden states from transformer model and computes the
    logits. In order to calculate the worthful representations from the transformer
    outputs, this class adopts time-based attention gating. Precisely, this class
    computes the importance of each word by using features and then apply
    weight-averaging to the features across the time axis.
    Since original classification models (i.e. `*ForSequenceClassification`) use simple
    feed-forward layers, the attention-based classification head can learn better
    generality.
    Args:
        hidden_size: The dimensionality of hidden units.
        num_labels: The number of labels to predict.
        dropout_prob: The dropout probability used in both attention and projection
            layers. Default is `0.1`.
    """

    def __init__(self, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Calculate the attention scores and apply the attention mask so that the
        # features of the padding tokens are not attended to the representation.
        attn = self.attention(features)
        if attention_mask is not None:
            attn += (1 - attention_mask.unsqueeze(-1)) * -10000.0

        # Pool the features across the timesteps and calculate logits.
        x = (features * attn.softmax(dim=1)).sum(dim=1)
        return x


class MeanPooling(nn.Module):
    """
    Average embeddings from last hidden state to get averaged/mean embeddings.
    Furthermore, this module doesn't take [PAD] tokens embeddings into account to get better representations.
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MaxPooling(nn.Module):
    """
    Take max across embeddings from last hidden state.
    """

    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        last_hidden_state[
            input_mask_expanded == 0
        ] = -1e9  # Set padding tokens to large negative value
        max_embeddings = torch.max(last_hidden_state, 1)[0]
        return max_embeddings


class Conv1DPooling(nn.Module):
    """
    Use conv1d layers to filter unwanted features and keep only important features across last hidden state
    """

    def __init__(self, in_size: int = 768, hidden_size: int = 512):
        super(Conv1DPooling, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.cnn1 = nn.Conv1d(self.in_size, self.hidden_size, kernel_size=2, padding=1)
        self.cnn2 = nn.Conv1d(self.hidden_size, 1, kernel_size=2, padding=1)

    def forward(self, last_hidden_state: torch.Tensor):
        last_hidden_state = last_hidden_state.permute(0, 2, 1)
        cnn_embeddings = F.relu(self.cnn1(last_hidden_state))
        cnn_embeddings = self.cnn2(cnn_embeddings)
        logits, _ = torch.max(cnn_embeddings, 2)

        return logits


class TransformerHead(nn.Module):
    def __init__(self, in_features: int, num_layers: int = 1, nhead: int = 8):
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=in_features, nhead=nhead),
            num_layers=num_layers,
        )
        self.row_fc = nn.Linear(in_features, 1)

    def forward(self, x):
        out = self.transformer(x)
        out = self.row_fc(out).squeeze(-1)
        return out


# https://www.kaggle.com/competitions/ubiquant-market-prediction/discussion/302977
class CorrLoss(nn.Module):
    """
    Use 1 - correlational coefficience between the output of the network and the target as the loss
    input (o, t):
        o: Variable of size (batch_size, 1) output of the network
        t: Variable of size (batch_size, 1) target value
    output (corr):
        corr: Variable of size (1)
    """

    def __init__(self):
        super(CorrLoss, self).__init__()

    def forward(self, o: torch.Tensor, t: torch.Tensor):
        assert o.size() == t.size()
        # calculate z-score for o and t
        o_m = o.mean(dim=0)
        o_s = o.std(dim=0)
        o_z = (o - o_m) / (o_s + 1e-7)

        t_m = t.mean(dim=0)
        t_s = t.std(dim=0)
        t_z = (t - t_m) / (t_s + 1e-7)

        # calculate corr between o and t
        tmp = o_z * t_z
        corr = tmp.mean(dim=0)
        return 1 - corr
