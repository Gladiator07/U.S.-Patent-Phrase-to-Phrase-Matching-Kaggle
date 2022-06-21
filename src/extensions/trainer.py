import torch.nn as nn
import wandb
from modeling.mixins import CorrLoss
from transformers import Trainer


class CustomLossTrainer(Trainer):
    """
    Trainer supporting `mse`, `bce` and `pearson` as loss functions
    """

    def __init__(self, *args, loss_type="mse", **kwargs):
        super().__init__(*args, **kwargs)
        # LOSS
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == "pearson":
            self.loss_fn = CorrLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss = self.loss_fn(logits.view(-1).to(labels.dtype), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class CustomWeightedClassificationTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):

        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        if wandb.run is not None:
            wandb.log({"val_logits": wandb.Histogram(eval_output.predictions)})

        return eval_output
