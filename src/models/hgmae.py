from itertools import chain
from functools import partial
import torch
import torch.nn as nn
from easydict import EasyDict
from torch_geometric.utils import dropout_edge, add_self_loops, to_torch_sparse_tensor
import torch.nn.functional as F

from utils.functions import sce_loss, sparse_sce_loss
from models.base import HAN


class HGMAE(nn.Module):
    def __init__(self, model_config, num_metapath=3, focused_feature_dim=768):
        super().__init__()

        self.num_metapath = num_metapath
        self.focused_feature_dim = focused_feature_dim

        self.target_node_type = "user"

        EncoderModelConfig = model_config.EncoderModelConfig
        DecoderModelConfig = model_config.DecoderModelConfig
        MPModelConfig = model_config.MPModelConfig
        assert EncoderModelConfig.hidden_dim % EncoderModelConfig.num_heads == 0
        assert DecoderModelConfig.hidden_dim % DecoderModelConfig.num_heads == 0
        assert EncoderModelConfig.input_dim == self.focused_feature_dim
        assert DecoderModelConfig.output_dim == self.focused_feature_dim

        self.EncoderModelClass = model_config.EncoderModelClass
        self.DecoderModelClass = model_config.DecoderModelClass

        if self.EncoderModelClass in ("GAT", "DotGAT", "HAN"):
            enc_hidden_dim = (
                EncoderModelConfig.hidden_dim // EncoderModelConfig.num_heads
            )
            enc_num_heads = EncoderModelConfig.num_heads
        else:
            enc_hidden_dim = EncoderModelConfig.hidden_dim
            enc_num_heads = 1

        # num head: decoder
        if self.DecoderModelClass in ("GAT", "DotGAT", "HAN"):
            dec_hidden_dim = (
                DecoderModelConfig.hidden_dim // DecoderModelConfig.num_heads
            )
            dec_num_heads = DecoderModelConfig.num_heads
        else:
            dec_hidden_dim = DecoderModelConfig.hidden_dim
            dec_num_heads = 1

        dec_in_dim = DecoderModelConfig.input_dim

        self.encoder = HAN(
            num_metapath=num_metapath,
            input_dim=EncoderModelConfig.input_dim,
            hidden_dim=enc_hidden_dim,
            output_dim=enc_hidden_dim,
            num_layers=EncoderModelConfig.num_layers,
            num_heads=enc_num_heads,
            num_output_heads=enc_num_heads,
            activation=EncoderModelConfig.activation,
            dropout=EncoderModelConfig.dropout,
            norm=create_norm(EncoderModelConfig.norm),
            encoding=True,
        )

        self.decoder = HAN(
            num_metapath=num_metapath,
            input_dim=DecoderModelConfig.input_dim,
            hidden_dim=dec_hidden_dim,
            output_dim=DecoderModelConfig.output_dim,
            num_layers=1,
            num_heads=enc_num_heads,
            num_output_heads=dec_num_heads,
            activation=DecoderModelConfig.activation,
            dropout=DecoderModelConfig.dropout,
            norm=create_norm(DecoderModelConfig.norm),
            encoding=False,
        )

        # type-specific attribute restoration
        self.attr_restoration_loss = partial(
            sce_loss, alpha=model_config.additional.alpha_l
        )
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.focused_feature_dim))
        self.encoder_to_decoder = nn.Linear(
            EncoderModelConfig.hidden_dim, DecoderModelConfig.input_dim, bias=False
        )
        self.replace_rate = model_config.additional.replace_rate
        self.leave_unchanged = model_config.additional.leave_unchanged
        assert (
            self.replace_rate + self.leave_unchanged < 1
        ), "Replace rate + leave_unchanged must be smaller than 1"

        # mp edge recon
        self.mp_edge_mask_rate = MPModelConfig.edge_mask_rate
        self.mp_edge_recon_loss = partial(
            sparse_sce_loss, alpha=MPModelConfig.edge_alpha_l
        )
        self.encoder_to_decoder_edge_recon = nn.Linear(
            dec_in_dim, dec_in_dim, bias=False
        )

        # mp2vec feat pred
        self.mps_embedding_dim = MPModelConfig.hidden_dim
        self.mp_feat_mask_rate = MPModelConfig.feat_mask_rate
        self.mp2vec_feat_drop = MPModelConfig.dropout
        self.mp2vec_feat_pred_loss = partial(
            sce_loss, alpha=MPModelConfig.feature_alpha_l
        )
        self.enc_out_to_mp2vec_feat_mapping = nn.Sequential(
            nn.Linear(dec_in_dim, self.mps_embedding_dim),
            nn.PReLU(),
            nn.Dropout(self.mp2vec_feat_drop),
            nn.Linear(self.mps_embedding_dim, self.mps_embedding_dim),
            nn.PReLU(),
            nn.Dropout(self.mp2vec_feat_drop),
            nn.Linear(self.mps_embedding_dim, self.mps_embedding_dim),
        )

    @property
    def output_hidden_dim(self):
        return self.hidden_dim

    def forward(self, batch):
        # prepare for mp2vec feat pred

        mp2vec_feat = batch[self.target_node_type].p_x
        origin_feat = batch[self.target_node_type].x

        metapath_dict = batch.metapath_dict

        mp_edge_index = [batch[mp_type].edge_index for mp_type in metapath_dict]
        # type-specific attribute restoration
        tar_loss, feat_recon, att_mp, enc_out, mask_nodes = self.mask_attr_restoration(
            origin_feat, mp_edge_index
        )

        # mp based edge reconstruction

        mer_loss = 0
        # mer_loss = self.mask_mp_edge_reconstruction(origin_feat, mp_edge_index)

        mp2vec_feat_pred = self.enc_out_to_mp2vec_feat_mapping(enc_out)

        pfp_loss = self.mp2vec_feat_pred_loss(mp2vec_feat_pred, mp2vec_feat)

        data_to_return = EasyDict(
            {
                "tar_loss": tar_loss,
                # "mer_loss": mer_loss,
                "pfp_loss": pfp_loss,
            }
        )

        return data_to_return

    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        perm_mask = torch.randperm(num_mask_nodes, device=x.device)
        num_leave_nodes = int(self.leave_unchanged * num_mask_nodes)
        num_noise_nodes = int(self.replace_rate * num_mask_nodes)
        num_real_mask_nodes = num_mask_nodes - num_leave_nodes - num_noise_nodes
        token_nodes = mask_nodes[perm_mask[:num_real_mask_nodes]]
        noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
        noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[
            :num_noise_nodes
        ]

        out_x = x.clone()
        out_x[token_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token
        if num_noise_nodes > 0:
            out_x[noise_nodes] = x[noise_to_be_chosen]

        return out_x, (mask_nodes, keep_nodes)

    def mask_attr_restoration(self, feat, gs):
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(
            feat, self.mp_feat_mask_rate
        )

        enc_out, _ = self.encoder(use_x, gs)

        # ---- attribute reconstruction ----
        enc_out_mapped = self.encoder_to_decoder(enc_out)
        if self.DecoderModelClass != "mlp":
            # re-mask
            enc_out_mapped[mask_nodes] = 0  # TODO: learnable? remove?

        if self.DecoderModelClass == "mlp":
            feat_recon = self.decoder(enc_out_mapped)
        else:
            feat_recon, att_mp = self.decoder(enc_out_mapped, gs)

        x_init = feat[mask_nodes]
        x_rec = feat_recon[mask_nodes]
        loss = self.attr_restoration_loss(x_rec, x_init)

        return loss, feat_recon, att_mp, enc_out, mask_nodes

    def mask_mp_edge_reconstruction(self, feat, mp_edge_index):
        masked_mp_edge_index = []
        for i in range(len(mp_edge_index)):
            masked_edge_index, _ = dropout_edge(
                mp_edge_index[i], p=self.mp_edge_mask_rate, training=self.training
            )
            masked_edge_index, _ = add_self_loops(
                edge_index=masked_edge_index, num_nodes=feat.shape[0]
            )
            masked_mp_edge_index.append(masked_edge_index)
        enc_rep, _ = self.encoder(feat, masked_mp_edge_index)
        rep = self.encoder_to_decoder_edge_recon(enc_rep)

        if self.DecoderModelClass == "mlp":
            feat_recon = self.decoder(rep)
        else:
            feat_recon, att_mp = self.decoder(rep, masked_mp_edge_index)

        gs_recon = torch.mm(feat_recon, feat_recon.T)

        loss = 0
        for i in range(len(mp_edge_index)):
            adj_dense = to_torch_sparse_tensor(
                mp_edge_index[i], size=feat.shape[0]
            ).to_dense()
            loss += att_mp[i] * self.mp_edge_recon_loss(gs_recon, adj_dense)
            # loss = att_mp[i] * self.mp_edge_recon_loss(gs_recon_only_masked_places_list[i], mps_only_masked_places_list[i])  # loss only on masked places
        return loss

    def get_embeds(self, batch):
        metapath_dict = batch.metapath_dict
        origin_feat = batch[self.target_node_type].x
        mp_edge_index = [batch[mp_type].edge_index for mp_type in metapath_dict]
        rep, _ = self.encoder(origin_feat, mp_edge_index)
        return rep.detach()

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return None


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = (
            torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        )
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(
            tensor
        )
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias
