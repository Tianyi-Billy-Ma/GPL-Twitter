import torch
import torch.nn.functional as F
import wandb
import numpy as np
from easydict import EasyDict

import logging

logger = logging.getLogger(__name__)
from trainers.base_executor import BaseExecutor

from models.base import BaseModel


class MLPExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        self.dataname = list(data_loader.data.keys())[0]

        self.target_node_type = self.config.train.additional.target_node_type
        ModelClass = globals()[self.config.model_config.ModelClass]
        self.model = ModelClass(
            config=config,
        )

        self.loss_fn = F.nll_loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.config.train.lr,
        )
        return {"optimizer": self.optimizer}

    def training_step(self, batch, batch_idx):
        target_nodes = batch[self.target_node_type]
        x, y = target_nodes.x, target_nodes.y
        mask = target_nodes.mask
        logit = self.model(x)
        batch_loss = self.loss_fn(logit[mask], y[mask])
        self.log(
            "train/batch_loss",
            batch_loss,
            prog_bar=True,
            # on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        data_to_return = EasyDict(
            {
                "loss": batch_loss,
            }
        )
        return data_to_return

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        target_nodes = batch[self.target_node_type]
        x, y = target_nodes.x, target_nodes.y
        mask = target_nodes.mask
        logit = self.model(x)
        batch_loss = self.loss_fn(logit[mask], y[mask])
        data_to_return = EasyDict(
            {
                "loss": batch_loss,
                "logit": logit[mask],
                "target": y[mask],
            }
        )
        return data_to_return

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        target_nodes = batch[self.target_node_type]
        x, y = target_nodes.x, target_nodes.y
        mask = target_nodes.mask
        logit = self.model(x)
        batch_loss = self.loss_fn(logit[mask], y[mask])
        data_to_return = EasyDict(
            {
                "loss": batch_loss,
                "logit": logit[mask],
                "target": y[mask],
            }
        )
        return data_to_return

    def evaluate_outputs(self, step_outputs, current_data_loader, dataset_name):
        pred_loss = np.mean([step_output.loss.item() for step_output in step_outputs])
        logits = torch.cat([step_output.logit for step_output in step_outputs], dim=0)
        y_true = torch.cat([step_output.target for step_output in step_outputs], dim=0)
        y_pred = F.log_softmax(input=logits, dim=1).argmax(dim=-1, keepdim=False)

        data_used_for_metrics = EasyDict(
            y_true=y_true.detach().cpu().numpy(),
            y_pred=y_pred.detach().cpu().numpy(),
        )
        log_dict = self.compute_metrics(data_used_for_metrics)
        log_dict["loss"] = pred_loss

        columns = ["user_id", "y_true", "y_pred"]
        test_table = wandb.Table(columns=columns)
        for i in range(len(y_true)):
            test_table.add_data(
                i,
                y_true[i].detach().cpu().numpy().item(),
                y_pred[i].detach().cpu().numpy().item(),
            )
        log_dict.artifacts.test_table = test_table

        return log_dict

    def logging_results(self, log_dict, prefix="test"):
        metrics_to_log = EasyDict()

        for metric, value in log_dict.metrics.items():
            metrics_to_log[f"{prefix}/{metric}"] = value

        metrics_to_log[f"{prefix}/loss"] = log_dict.loss
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
        wandb_artifacts_to_log = dict()
        wandb_artifacts_to_log.update(
            {
                f"predictions/epoch_{self.current_epoch}_MODE_{self.config.mode}_SET_{prefix}": log_dict.artifacts[
                    "test_table"
                ]
            }
        )

        if self.config.args.log_prediction_tables:
            self.wandb_logger.experiment.log(wandb_artifacts_to_log, commit=False)
