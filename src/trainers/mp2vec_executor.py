import pytorch_lightning as pl
import torch

from trainers.metric_processors import MetricsProcessor
from trainers.base_executor import BaseExecutor

from models.MetaPath2Vec import MP2Vec


class MP2VecExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        ModelClass = globals()[self.config.model_config.ModelClass]
        dataname = list(data_loader.data.keys())[0]
        self.model = ModelClass(
            config=config,
            hetero_graph=data_loader.data[dataname],
        )

    def configure_optimizers(self):
        self.optimizer = torch.optim.SparseAdam(
            list(self.model.parameters()),
            lr=self.config.train.lr,
        )
        return {"optimizer": self.optimizer}

    def train_dataloader(self):
        return self.model.loader(batch_size=self.config.train.batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        return self.model(batch)
