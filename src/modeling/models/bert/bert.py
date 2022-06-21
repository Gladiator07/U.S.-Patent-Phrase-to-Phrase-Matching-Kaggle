"""
Modified from https://github.com/huggingface/transformers/blob/a22db885b41b3a1b302fc206312ee4d99cdf4b7c/src/transformers/models/bert/modeling_bert.py#L1508

This class will be mainly used for anferico/bert-for-patents checkpoint (https://huggingface.co/anferico/bert-for-patents)
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from modeling.mixins import CorrLoss, AttentionHead


class BertForSequenceClassificationGeneral(BertPreTrainedModel):
    def __init__(
        self,
        config,
        loss_type: str = "mse",
        multi_sample_dropout: bool = False,
        attention_pool: bool = False,
    ):
        super().__init__(config)
        num_labels = getattr(config, "num_labels", 2)
        self.attention_pool = attention_pool

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )

        # POOL
        if self.attention_pool:
            # Attention pool
            self.attention_head = AttentionHead(
                config.hidden_size, dropout_prob=classifier_dropout
            )
            self.output = nn.Linear(config.hidden_size, num_labels)
        else:
            self.output = nn.Linear(config.hidden_size, num_labels)

        # DROPOUT
        if multi_sample_dropout:
            self.dropouts = nn.ModuleList([nn.Dropout(0.1 * (1 + d)) for d in range(5)])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(classifier_dropout)])

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

        outputs = self.bert(
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

        if self.attention_pool:
            pooled_output = self.attention_head(outputs[0], attention_mask)
        else:
            # CLS token pool
            pooled_output = outputs[1]

        # multi-sample dropout
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.output(dropout(pooled_output))
            else:
                logits += self.output(dropout(pooled_output))

        logits /= len(self.dropouts)
        loss = None
        if labels is not None:
            # loss function
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
