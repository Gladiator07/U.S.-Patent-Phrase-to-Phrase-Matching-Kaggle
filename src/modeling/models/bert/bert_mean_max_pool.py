"""
Modified from https://github.com/huggingface/transformers/blob/a22db885b41b3a1b302fc206312ee4d99cdf4b7c/src/transformers/models/bert/modeling_bert.py#L1508

This class will be mainly used for anferico/bert-for-patents checkpoint (https://huggingface.co/anferico/bert-for-patents)
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from modeling.mixins import CorrLoss, MaxPooling, MeanPooling


class BertMeanMaxPooling(BertPreTrainedModel):
    def __init__(
        self,
        config,
        loss_type: str = "mse",
    ):
        super().__init__(config)
        num_labels = getattr(config, "num_labels", 2)

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )

        self.pooler1 = MeanPooling()
        self.pooler2 = MaxPooling()
        self.dropout = nn.Dropout(classifier_dropout)
        self.output = nn.Linear(config.hidden_size * 2, num_labels)

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

        last_hidden_state = outputs[0]
        # mean max pooling
        mean_pooled_output = self.pooler1(last_hidden_state, attention_mask)
        max_pooled_output = self.pooler2(last_hidden_state, attention_mask)
        pooled_output = torch.cat((mean_pooled_output, max_pooled_output), 1)
        logits = self.output(self.dropout(pooled_output))
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
