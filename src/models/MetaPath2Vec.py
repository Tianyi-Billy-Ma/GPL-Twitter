import pytorch_lightning as pl

from torch_geometric.nn import MetaPath2Vec

from easydict import EasyDict

from models.MLP import MLP


class MP2Vec(pl.LightningModule):
    def __init__(self, config, hetero_graph):
        super().__init__()
        # self.config = config

        EncoderModelClass = config.model_config.EncoderModelClass
        EncoderModelConfig = config.model_config.EncoderModelConfig

        DecoderModelClass = config.model_config.DecoderModelClass
        DecoderModelConfig = config.model_config.DecoderModelConfig

        self.target_node_type = config.train.additional.target_node_type

        ModelConfig = EasyDict(EncoderModelConfig)
        edge_index_dict = hetero_graph.edge_index_dict
        if not "metapath" in ModelConfig:
            ModelConfig.metapath = hetero_graph.edge_types
        else:
            ModelConfig.metapath = [
                (src, rel, dst) for (src, rel, dst) in ModelConfig.metapath
            ]

        ModelConfig.num_nodes_dict = hetero_graph.num_nodes_dict
        self.encoder = globals()[EncoderModelClass](
            edge_index_dict=edge_index_dict, **ModelConfig
        )
        self.decoder = globals()[DecoderModelClass](**DecoderModelConfig)

    def loader(self, batch_size, shuffle):
        return self.encoder.loader(batch_size=batch_size, shuffle=shuffle)

    def forward(self, batch):
        pos_rw, neg_rw = batch
        loss = self.encoder.loss(pos_rw, neg_rw)
        return EasyDict(
            loss=loss,
        )

    def get_prediction(self):
        embs = self.get_embeddings()
        embs = self.decoder(embs)
        return embs

    def get_embeddings(self, node_type=None):
        if node_type is None:
            node_type = self.target_node_type
        return self.encoder(node_type).detach()
