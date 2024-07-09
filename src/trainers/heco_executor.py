import torch
import pytorch_lightning as pl


import logging

logger = logging.getLogger(__name__)


from trainers.base_executor import BaseExecutor
from models.heco import HeCo


class HeCoExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        self.dataname = list(data_loader.data.keys())[0]
        self.target_node_type = self.config.train.additional.target_node_type

        ModelClass = globals()[self.config.model_config.ModelClass]

        model_config = self.config.model_config
        self.model = ModelClass(model_config)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.model.parameters()), lr=self.config.train.lr
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        return {"loss": loss}
