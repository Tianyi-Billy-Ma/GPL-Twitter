import pytorch_lightning as pl
import torch
import os.path as osp
import numpy as np
import wandb
from easydict import EasyDict
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
import torch.nn as nn

from trainers.base_executor import BaseExecutor


from models.hgmae import HGMAE
from models.base import LogReg


class HGMAEExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        self.dataname = list(data_loader.data.keys())[0]
        self.target_node_type = self.config.train.additional.target_node_type

        ModelClass = globals()[self.config.model_config.ModelClass]

        model_config = self.config.model_config
        self.model = ModelClass(model_config)
        self.classifier = LogReg(**self.config.model_config.ClassifierModelConfig)
        self.loss_weights = EasyDict(self.config.model_config.loss_weights)
        self.loss_fn = F.nll_loss

        self.automatic_optimization = False

    def configure_optimizers(self):
        model_optimizer = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.config.train.lr,
        )
        classifier_optimizer = torch.optim.Adam(
            list(self.classifier.parameters()),
            lr=self.config.train.lr,
            weight_decay=self.config.train.wd,
        )
        return [model_optimizer, classifier_optimizer]

    def train_dataloader(self):
        self.train_dataloader_names = list(
            self.data_loader.data_loaders["train"].keys()
        )

        # TODO: we only allow one train data loader at the moment
        return self.train_dataloaders[0]

    def val_dataloader(self):
        self.val_dataloader_names = list(self.data_loader.data_loaders["valid"].keys())

        return self.valid_dataloaders

    def test_dataloader(self):
        self.test_dataloader_names = list(self.data_loader.data_loaders["test"].keys())

        return self.test_dataloaders

    def training_step(self, batch, batch_idx):
        model_optimizer, classifier_optimizer = self.optimizers()
        data_to_log = EasyDict()
        loss_dict = self.model(batch)

        total_loss = 0
        for loss_name, loss_val in loss_dict.items():
            data_to_log[f"train/{loss_name}"] = loss_val.item()
            total_loss += loss_val * self.loss_weights[f"{loss_name}_weight"]

        data_to_log["train/total_loss"] = total_loss.item()

        model_optimizer.zero_grad()
        self.manual_backward(total_loss)
        model_optimizer.step()

        mask = batch[self.target_node_type].mask
        y_true = batch[self.target_node_type].y
        embs = self.model.get_embeds(batch)
        embs = embs[self.target_node_type]
        output = self.classifier(embs)
        logits = F.log_softmax(output, dim=1)
        loss = self.loss_fn(logits[mask], y_true[mask])
        data_to_log["train/classifier_loss"] = loss.item()

        classifier_optimizer.zero_grad()
        self.manual_backward(loss)
        classifier_optimizer.step()

        self.log_dict(data_to_log, prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._compute_logit(batch, batch_idx, dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._compute_logit(batch, batch_idx, dataloader_idx)

    def _compute_logit(self, batch, batch_idx, dataloader_idx):
        loss_dict = self.model(batch)
        embs = self.model.get_embeds(batch)
        embs = embs[self.target_node_type]
        embs = self.classifier(embs)
        data_to_return = EasyDict()
        total_loss = 0
        for loss_name, loss_val in loss_dict.items():
            data_to_return[loss_name] = loss_val.item()
            total_loss += loss_val * self.loss_weights[f"{loss_name}_weight"]
        data_to_return["total_loss"] = total_loss.item()

        mask = batch[self.target_node_type].mask
        y_true = batch[self.target_node_type].y

        logits = F.log_softmax(embs, dim=1)
        pred_loss = self.loss_fn(logits[mask], y_true[mask])
        y_pred = torch.argmax(logits, dim=1)

        data_to_return["pred_loss"] = pred_loss.item()
        data_to_return["y_true"] = y_true.detach().cpu().numpy()
        data_to_return["y_pred"] = y_pred.detach().cpu().numpy()

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
                    logger=True,
                    sync_dist=True,
                )
            else:
                logger.info(f"{metric} is not a type that can be logged, skippped.")
