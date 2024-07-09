import pytorch_lightning as pl
import torch
import os.path as osp
import numpy as np
import wandb
from easydict import EasyDict
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

from trainers.metric_processors import MetricsProcessor
from trainers.base_executor import BaseExecutor


from models.MetaPath2Vec import MP2Vec


class MP2VecExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        ModelClass = globals()[self.config.model_config.ModelClass]
        self.dataname = list(data_loader.data.keys())[0]

        self.target_node_type = self.config.train.additional.target_node_type

        self.model = ModelClass(
            config=config,
            hetero_graph=data_loader.data[self.dataname],
        )

        for mode in self.data_loader.data_loaders.keys():
            tmp_dataloaders = []
            for dataset_name in data_loader.data_loaders[mode].keys():
                current_data_loader = self.model.loader(
                    batch_size=self.config[mode].batch_size,
                    shuffle=True,
                )
                current_data_loader.y = data_loader.data[self.dataname][
                    self.target_node_type
                ].y
                current_data_loader.mask = data_loader.data_loaders[mode][dataset_name][
                    self.target_node_type
                ].mask
                tmp_dataloaders.append(current_data_loader)
            setattr(
                self,
                f"{mode}_dataloaders",
                tmp_dataloaders,
            )
        self.loss_fn = F.nll_loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.SparseAdam(
            list(self.model.encoder.parameters()),
            lr=self.config.train.lr,
        )
        return {"optimizer": self.optimizer}

    def training_step(self, batch, batch_idx):
        batch_loss = self.model(batch).loss
        self.log(
            "train/batch_loss",
            batch_loss,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        data_to_return = {
            "loss": batch_loss,
        }
        return data_to_return

    def on_train_epoch_end(self):
        optimizer = torch.optim.Adam(
            list(self.model.decoder.parameters()),
            lr=self.config.train.lr,
            weight_decay=self.config.train.wd,
        )
        optimizer.zero_grad()
        current_data_loader = self.train_dataloader()
        y_true = current_data_loader.y.to(self.device)
        train_mask = current_data_loader.mask.to(self.device)
        output = self.model.get_prediction()
        logits = F.log_softmax(output, dim=1)
        pred_loss = self.loss_fn(logits[train_mask], y_true[train_mask])
        pred_loss.backward()
        optimizer.step()
        self.log(
            "train/pred_loss",
            pred_loss.item(),
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch_loss = self.model(batch).loss
        return EasyDict(
            {
                "batch_loss": batch_loss.detach().cpu().item(),
            }
        )

    def evaluate_outputs(self, step_outputs, current_data_loader, dataset_name):
        total_loss = np.mean([step_output.batch_loss for step_output in step_outputs])
        mask = current_data_loader.mask.to(self.device)
        y_true = current_data_loader.y.to(self.device)
        output = self.model.get_prediction()
        logits = F.log_softmax(output, dim=1)
        pred_loss = self.loss_fn(logits[mask], y_true[mask])
        y_pred = logits.argmax(dim=-1, keepdim=False)

        data_used_for_metrics = EasyDict(
            y_true=y_true[mask].detach().cpu().numpy(),
            y_pred=y_pred[mask].detach().cpu().numpy(),
        )
        log_dict = self.compute_metrics(data_used_for_metrics)
        log_dict["pred_loss"] = pred_loss.item()
        log_dict["total_loss"] = total_loss

        columns = ["user_id", "y_true", "y_pred"]
        test_table = wandb.Table(columns=columns)
        for i in range(len(data_used_for_metrics.y_true)):
            test_table.add_data(
                i,
                y_true[mask][i].detach().cpu().numpy().item(),
                y_pred[mask][i].detach().cpu().numpy().item(),
            )
        log_dict.artifacts.test_table = test_table
        return log_dict

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        batch_loss = self.model(batch).loss
        return EasyDict(
            {
                "batch_loss": batch_loss.detach().cpu().item(),
            }
        )

    def logging_results(self, log_dict, prefix):
        metrics_to_log = EasyDict()
        for metric, value in log_dict.metrics.items():
            metrics_to_log[f"{prefix}/{metric}"] = value

        metrics_to_log[f"{prefix}/pred_loss"] = log_dict.pred_loss
        metrics_to_log[f"{prefix}/total_loss"] = log_dict.total_loss
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

    def on_train_end(self):
        save_embeddings = self.model.get_embeddings(self.target_node_type)
        save_path = osp.join(
            self.config.imgs_path,
            f"epoch_{self.current_epoch}_{self.target_node_type}.pt",
        )
        torch.save(save_embeddings, save_path)
        logger.info(f"Embeddings saved to: {save_path}")
        super().on_train_end()

    def on_validation_epoch_start(self):
        self.validation_step_outputs = [[] for _ in range(len(self.val_dataloader()))]

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.validation_step_outputs[dataloader_idx].append(outputs)

    def on_test_epoch_start(self):
        self.test_step_outputs = [[] for _ in range(len(self.test_dataloader()))]

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.test_step_outputs[dataloader_idx].append(outputs)
