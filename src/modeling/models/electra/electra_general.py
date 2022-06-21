from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from modeling.mixins import AttentionHead, CorrLoss


class ElectraForSequenceClassificationGeneral(ElectraPreTrainedModel):
    def __init__(
        self,
        config,
        loss_type: str = "mse",
        multi_sample_dropout: bool = False,
        attention_pool: bool = False,
    ):
        super().__init__(config)

        num_labels = 1

        self.electra = ElectraModel(config)
        self.attention_pool = attention_pool

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        # POOL
        if self.attention_pool:
            # Attention pool
            self.attention_head = AttentionHead(
                config.hidden_size, dropout_prob=drop_out
            )

        self.output = nn.Linear(config.hidden_size, num_labels)

        # DROPOUT
        if multi_sample_dropout:
            self.dropouts = nn.ModuleList([nn.Dropout(0.1 * (1 + d)) for d in range(5)])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(drop_out)])

        # LOSS
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == "pearson":
            self.loss_fn = CorrLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = discriminator_hidden_states[0]

        if self.attention_pool:
            pooled_output = self.attention_head(sequence_output, attention_mask)
        else:
            pooled_output = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS])

        # multi-sample dropout
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.output(dropout(pooled_output))
            else:
                logits += self.output(dropout(pooled_output))

        logits /= len(self.dropouts)

        loss = None
        if labels is not None:
            logits = logits.view(-1).to(labels.dtype)
            loss = self.loss_fn(logits, labels.view(-1))

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
