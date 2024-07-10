import torch
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict

import logging

logger = logging.getLogger(__name__)
from trainers.base_executor import BaseExecutor
from models.MLP import MLP


class ReWeightExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        self.dataname = list(data_loader.data.keys())[0]
        self.target_node_type = self.config.train.additional.target_node_type

        ModelClass = self.config.model_config.ModelClass
        ModelClassConfig = self.config.model_config.ModelClassConfig

        self.model = globals()[ModelClass](**ModelClassConfig)

        self.LossClass = self.config.model_config.LossClass
        self.LossClassConfig = self.config.model_config.LossClassConfig

    def configure_optimizers(self):
        model_optimizer = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.config.train.lr,
        )
        return model_optimizer

    def training_step(self, batch, batch_idx):
        input_features = batch[self.target_node_type].x
        mask = batch[self.target_node_type].mask
        y_true = batch[self.target_node_type].y

        no_of_classes = y_true.unique().shape[0]
        samples_per_cls = np.bincount(y_true[mask].cpu().numpy())

        logits = self.model(input_features)

        loss = CB_loss(
            labels=y_true[mask],
            logits=logits[mask],
            samples_per_cls=samples_per_cls,
            no_of_classes=no_of_classes,
            loss_type=self.LossClass,
            device=self.device,
            **self.LossClassConfig,
        )
        data_to_return = {"loss": loss}
        self.log(
            "train/loss",
            loss.item(),
            # on_step=False,
            # on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return data_to_return

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._compute_logit(batch, batch_idx, dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._compute_logit(batch, batch_idx, dataloader_idx)

    def _compute_logit(self, batch, batch_idx, dataloader_idx):
        input_features = batch[self.target_node_type].x
        mask = batch[self.target_node_type].mask
        y_true = batch[self.target_node_type].y
        data_to_return = EasyDict()

        no_of_classes = y_true.unique().shape[0]
        samples_per_cls = np.bincount(y_true[mask].cpu().numpy())

        logits = self.model(input_features)
        y_pred = torch.argmax(logits, dim=1)
        loss = CB_loss(
            labels=y_true[mask],
            logits=logits[mask],
            samples_per_cls=samples_per_cls,
            no_of_classes=no_of_classes,
            loss_type=self.LossClass,
            device=self.device,
            **self.LossClassConfig,
        )
        data_to_return["pred_loss"] = loss.item()
        data_to_return["y_true"] = y_true[mask].detach().cpu().numpy()
        data_to_return["y_pred"] = y_pred[mask].detach().cpu().numpy()

        return data_to_return

    def evaluate_outputs(self, step_outputs, current_data_loader, dataset_name):
        data_used_for_metrics = EasyDict(
            y_true=step_outputs.y_true,
            y_pred=step_outputs.y_pred,
        )
        log_dict = self.compute_metrics(data_used_for_metrics)

        for key, val in step_outputs.items():
            if key.endswith("loss"):
                log_dict[key] = val

        return log_dict

    def logging_results(self, log_dict, prefix):
        metrics_to_log = EasyDict()

        for metric, value in log_dict.metrics.items():
            metrics_to_log[f"{prefix}/{metric}"] = value
        metrics_to_log[f"{prefix}/epoch"] = self.current_epoch

        logger.info(
            f"Evaluation results [{self.trainer.state.stage}]: {metrics_to_log}"
        )
        if self.trainer.state.stage in ["sanity_check"]:
            logging.warning("Sanity check mode, not saving to loggers.")
            return
        for metric, value in metrics_to_log.items():
            if type(value) in [float, int, np.float64]:
                self.log(
                    metric,
                    float(value),
                    # on_step=False,
                    # on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )
            else:
                logger.info(f"{metric} is not a type that can be logged, skippped.")


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(
        input=logits, target=labels, reduction="none"
    )

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(
            -gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits))
        )

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


def CB_loss(
    labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, device
):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float().to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot.to(device)
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(
            input=logits, target=labels_one_hot, weights=weights
        )
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(
            input=pred, target=labels_one_hot, weight=weights
        )
    return cb_loss
