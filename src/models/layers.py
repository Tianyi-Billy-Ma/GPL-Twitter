import torch

import torch.nn as nn

from torch_geometric.nn import GATConv


class SemanticAttention(nn.Module):
    def __init__(self, input_dim, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        out_emb = (beta * z).sum(1)  # (N, D * K)
        att_mp = beta.mean(0).squeeze()

        return out_emb, att_mp


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(
        self,
        num_metapath,
        in_dim,
        out_dim,
        nhead,
        dropout,
        activation,
        norm,
    ):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        if norm:
            self.norm = norm(out_dim * nhead)
        else:
            self.norm = nn.Identity()
        self.activation = activation
        self.layers = nn.ModuleList()
        for _ in range(num_metapath):
            self.layers.append(
                GATConv(
                    in_dim,
                    out_dim,
                    heads=nhead,
                    dropout=dropout,
                )
            )
        self.semantic_attention = SemanticAttention(input_dim=out_dim * nhead)

    def forward(self, x, mp_edge_index):
        semantic_embeddings = []

        for i, edge_index in enumerate(mp_edge_index):
            emb = self.layers[i](x, edge_index)
            emb = self.norm(emb)
            emb = self.activation(emb)

            semantic_embeddings.append(emb.flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        out, att_mp = self.semantic_attention(semantic_embeddings)  # (N, D * K)

        return out, att_mp
