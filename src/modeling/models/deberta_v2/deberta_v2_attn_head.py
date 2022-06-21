"""
Modified from https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/deberta_v2/modeling_deberta_v2.py
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    DebertaV2Model,
    DebertaV2PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from modeling.mixins import CorrLoss, TransformerHead


class DebertaV2ForSequenceClassificationAttnHead(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(
        self,
        config,
        loss_type: str = "mse",
    ):
        super().__init__(config)

        config.update({"output_hidden_states": True})
        num_labels = getattr(config, "num_labels", 2)

        self.deberta = DebertaV2Model(config)

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        self.transformer_head = TransformerHead(
            in_features=config.hidden_size, num_layers=1, nhead=8
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # infer input size from `self.transformer_head`
        self.output = nn.LazyLinear(num_labels)

        # LOSS
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == "pearson":
            self.loss_fn = CorrLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_states = outputs[0]

        features = self.transformer_head(last_hidden_states)
        logits = self.output(self.dropout(features))

        loss = None
        if labels is not None:
            logits = logits.view(-1).to(labels.dtype)
            loss = self.loss_fn(logits, labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
