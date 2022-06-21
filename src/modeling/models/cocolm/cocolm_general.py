import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

from cocolm.modeling_cocolm import COCOLMModel, COCOLMPreTrainedModel
from modeling.mixins import AttentionHead, CorrLoss


class COCOLMForSequenceClassificationGeneral(COCOLMPreTrainedModel):
    def __init__(
        self,
        config,
        loss_type: str = "mse",
        multi_sample_dropout: bool = False,
        attention_pool: bool = False,
    ):
        super().__init__(config)

        config.num_labels = 1
        drop_out = config.cls_dropout_prob

        self.attention_pool = attention_pool
        self.cocolm = COCOLMModel(config)

        # POOL
        if self.attention_pool:
            # Attention pool
            self.attention_head = AttentionHead(
                config.hidden_size, dropout_prob=drop_out
            )

        self.output = nn.Linear(config.hidden_size, config.num_labels)

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

        if self.attention_pool:
            pooled_output = self.attention_head(outputs[0], attention_mask)
        else:
            pooled_output = outputs[0][:, 0, :]  # take <s> token (equiv. to [CLS])

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
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs,
        )
