import pytorch_lightning as pl
import torch
import os.path as osp
import numpy as np
import wandb
from easydict import EasyDict
import torch.nn.functional as F
import logging

from trainers.base_executor import BaseExecutor


from models.hgmae import HGMAE


class HGMAEExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        self.dataname = list(data_loader.data.keys())[0]
        self.target_node_type = self.config.train.additional.target_node_type

        ModelClass = globals()[self.config.model_config.ModelClass]

        num_metapath = len(data_loader.data[self.dataname].metapath_dict)
        focused_feature_dim = data_loader.data[self.dataname][
            self.target_node_type
        ].x.shape[1]

        self.model = ModelClass(config, num_metapath, focused_feature_dim)
        self.loss_weights = EasyDict(self.config.model_config.loss_weights)

    def configure_optimizers(self):
        self.optimizer = torch.optim.SparseAdam(
            list(self.model.encoder.parameters()),
            lr=self.config.train.lr,
        )
        return {"optimizer": self.optimizer}

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
        loss_dict = self.model(batch)

        total_loss = 0
        for loss_name, loss_val in loss_dict.items():
            self.log(
                f"train/{loss_name}",
                loss_val.item(),
                prog_bar=True,
                # on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            total_loss += loss_val * self.loss_weights[f"{loss_name}_weight"]

        self.log(
            "train/total_loss",
            total_loss.item(),
            prog_bar=True,
            # on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        data_to_return = {
            "loss": total_loss,
        }
        return data_to_return
