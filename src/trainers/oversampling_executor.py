import torch
import pytorch_lightning as pl


import logging

logger = logging.getLogger(__name__)
import torch.nn.functional as F
from easydict import EasyDict
import numpy as np

from trainers.base_executor import BaseExecutor
from models.heco import HeCo
from models.base import LogReg
from models.smote import SMOTE
from models.MLP import MLP
from models.oversampling import OverSampling


class OverSamplingExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.dataname = list(data_loader.data.keys())[0]
        self.target_node_type = self.config.train.additional.target_node_type

        OverSamplingClass = self.config.model_config.OverSamplingClass

        self.oversampling = globals()[OverSamplingClass]()

        ModelClass = self.config.model_config.ModelClass
        ModelClassConfig = self.config.model_config.ModelClassConfig

        self.model = globals()[ModelClass](**ModelClassConfig)

        self.loss_fn = F.nll_loss

    def configure_optimizers(self):
        classifier_optimizer = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.config.train.lr,
            weight_decay=self.config.train.wd,
        )
        return {"optimizer": classifier_optimizer}

    def training_step(self, batch, batch_idx):
        input_features = batch[self.target_node_type].x
        mask = batch[self.target_node_type].mask
        y_true = batch[self.target_node_type].y

        sampled_features, sampled_labels = self.oversampling(
            input_features[mask], y_true[mask]
        )

        logits = self.model(sampled_features)
        logits = F.log_softmax(logits, dim=1)
        loss = self.loss_fn(logits, sampled_labels.long())

        data_to_return = {"loss": loss}
        self.log(
            "train/loss",
            loss.item(),
            # on_step=False,
            # on_epoch=True,
            prog_bar=True,
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

        logits = self.model(input_features)
        y_pred = torch.argmax(logits, dim=1)

        logits = F.log_softmax(logits, dim=1)
        loss = self.loss_fn(logits[mask], y_true[mask])

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
