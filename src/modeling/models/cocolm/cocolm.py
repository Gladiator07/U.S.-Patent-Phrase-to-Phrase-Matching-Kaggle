import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from cocolm.modeling_cocolm import (
    COCOLMPreTrainedModel,
    COCOLMClassificationHead,
    COCOLMModel,
)
from modeling.mixins import CorrLoss


class COCOLMForSequenceClassification(COCOLMPreTrainedModel):
    def __init__(self, config, loss_type: str = "mse"):
        super().__init__(config)
        self.config = config
        config.num_labels = 1

        self.cocolm = COCOLMModel(config)
        self.classifier = COCOLMClassificationHead(config)
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

        logits = self.classifier(outputs[0])
        loss = None
        if labels is not None:
            logits = logits.view(-1).to(labels.dtype)
            loss = self.loss_fn(logits, labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs,
        )
