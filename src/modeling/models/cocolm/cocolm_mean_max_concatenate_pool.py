import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

from cocolm.modeling_cocolm import COCOLMModel, COCOLMPreTrainedModel
from modeling.mixins import CorrLoss, MaxPooling, MeanPooling


class COCOLMMeanMaxPool(COCOLMPreTrainedModel):
    def __init__(self, config, loss_type: str = "mse"):
        super().__init__(config)

        config.num_labels = 1
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        self.cocolm = COCOLMModel(config)
        self.pooler1 = MeanPooling()
        self.pooler2 = MaxPooling()
        self.dropout = nn.Dropout(drop_out)
        self.output = nn.Linear(config.hidden_size * 2, config.num_labels)

        # LOSS
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == "pearson":
            self.loss_fn = CorrLoss()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.cocolm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        # mean max pooling
        mean_pooled_output = self.pooler1(outputs[0], attention_mask)
        max_pooled_output = self.pooler2(outputs[0], attention_mask)
        pooled_output = torch.cat((mean_pooled_output, max_pooled_output), 1)
        logits = self.output(self.dropout(pooled_output))

        loss = None
        if labels is not None:
            logits = logits.view(-1).to(labels.dtype)
            loss = self.loss_fn(logits, labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs,
        )
