import pytorch_lightning as pl
import torch

from trainers.base_executor import BaseExecutor

from models.hgmae import HGMAE


class iHGTExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        self.dataname = list(data_loader.data.keys())[0]
        self.target_node_type = self.config.train.additional.target_node_type

        PretrainModelClass = config.model_config.PretrainModelClass
        PretrainModelConfig = config.model_config.PretrainModelConfig
        PretrainModelCkptPath = config.model_config.PretrainModelCkptPath
        PretrainModelName = config.model_config.PretrainModelName
        PretrainModelCkpt = torch.load(PretrainModelCkptPath)
        PretrainModelWeights = {
            k[len(PretrainModelName) + 1 :]: v
            for k, v in PretrainModelCkpt["state_dict"].items()
            if k.startswith(f"{PretrainModelName}.")
        }
        self.pretrain_model = globals()[PretrainModelClass](PretrainModelConfig)
        self.pretrain_model.load_state_dict(PretrainModelWeights)
        print()

    def configure_optimizers(self):
        model_optimizer = torch.optim.Adam(
            list(self.model.encoder.parameters()),
            lr=self.config.train.lr,
        )
        return model_optimizer

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
        pass
