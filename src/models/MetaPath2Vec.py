import pytorch_lightning as pl

from torch_geometric.nn import MetaPath2Vec

from easydict import EasyDict


class MP2Vec(pl.LightningModule):
    def __init__(self, config, hetero_graph):
        super().__init__()
        # self.config = config
        ModelConfig = config.model_config.ModelConfig

        ModelConfig = EasyDict(ModelConfig)
        edge_index_dict = hetero_graph.edge_index_dict
        if not "metapath" not in ModelConfig:
            ModelConfig.metapath = hetero_graph.edge_types
        self.model = MetaPath2Vec(edge_index_dict=edge_index_dict, **ModelConfig)
        print()

    def loader(self, batch_size, shuffle):
        return self.model.loader(batch_size=batch_size, shuffle=shuffle)

    def froward(self, batch):
        return self.model.loss(batch)
