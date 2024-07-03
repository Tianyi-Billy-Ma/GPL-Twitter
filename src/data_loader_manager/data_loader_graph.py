import sys

sys.dont_write_bytecode = True

import os
import os.path as osp

import logging

logger = logging.getLogger(__name__)

import functools
from easydict import EasyDict
import pandas as pd
import json

from collections import defaultdict

import torch
import torch_geometric as pyg
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import (
    WikiCS,
    Planetoid,
    WordNet18RR,
    FB15k_237,
    Coauthor,
    Amazon,
    WebKB,
    WikipediaNetwork,
    AMiner,
    DBLP,
)
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

from data_loader_manager.data_loader_wrapper import DataLoaderWrapper

from utils.dirs import load_file
from utils.functions import create_split_masks


class DataLoaderForGraph(DataLoaderWrapper):
    def __init__(self, config):
        DataLoaderWrapper.__init__(self, config)

    def LoadTwitterData(self, module_config):
        data_path = osp.join(self.config.DATA_FOLDER, module_config.config.path)
        save_or_load_path = osp.join(
            data_path, "processed", f"{module_config.config.name}.pt"
        )
        data_dict = EasyDict({})
        if osp.exists(save_or_load_path) and module_config.option == "default":
            data = torch.load(save_or_load_path)
        else:
            os.makedirs(osp.join(data_path, "processed"), exist_ok=True)
            raw_data_path = osp.join(data_path, "raw")

            raw_edges = load_file(osp.join(raw_data_path, "edges.json"))
            keyword_dict = load_file(osp.join(raw_data_path, "keyword_embeddings.pkl"))
            tweet_dict = load_file(osp.join(raw_data_path, "tweet_embeddings.pkl"))
            user_dict = load_file(osp.join(raw_data_path, "user_embeddings.pkl"))
            user_labels = load_file(osp.join(raw_data_path, "labels.json"))

            user_ids, user_embeddings = user_dict["ids"], user_dict["embeddings"]
            tweet_ids, tweet_embeddings = tweet_dict["ids"], tweet_dict["embeddings"]
            keyword_ids, keyword_embeddings = (
                keyword_dict["ids"],
                keyword_dict["embeddings"],
            )
            ### ID Mapping ###
            id_mapping = {}
            count_dict = {"user": 0, "tweet": 0, "keyword": 0}
            for id in user_ids + tweet_ids + keyword_ids:
                node_type = id.split("_")[0]
                id_mapping[id] = count_dict[node_type]
                count_dict[node_type] += 1

            ### Preprocess the labels ###
            label_mapping = {"Negative": 0, "Seller": 1, "Buyer": 2, "Related": 3}
            labels = []
            for user_id, label in user_labels.items():
                assert id_mapping[user_id] == len(labels)
                labels.append(label_mapping[label])

            ### Preprocess the edges ###
            relations = defaultdict(set)

            if "build_metapath_for_MetaPath2Vec" in module_config.config.preprocess:
                for edge in raw_edges:
                    src_node_id = id_mapping[edge["source_id"]]
                    tar_node_id = id_mapping[edge["target_id"]]
                    relation_type = edge["relation"]
                    src_node_type = edge["source_id"].split("_")[0]
                    tar_node_type = edge["target_id"].split("_")[0]
                    triple = relation_type.split("-")
                    assert src_node_type == triple[0] and tar_node_type == triple[2]

                    relations[(src_node_type, "to", tar_node_type)].add(
                        (src_node_id, tar_node_id)
                    )
                    relations[(tar_node_type, "to", src_node_type)].add(
                        (tar_node_id, src_node_id)
                    )
            else:
                for edge in raw_edges:
                    src_node_id = id_mapping[edge["source_id"]]
                    tar_node_id = id_mapping[edge["target_id"]]
                    relation_type = edge["relation"]
                    src_node_type = edge["source_id"].split("_")[0]
                    tar_node_type = edge["target_id"].split("_")[0]
                    triple = relation_type.split("-")
                    assert src_node_type == triple[0] and tar_node_type == triple[2]

                    relations[(src_node_type, triple[1], tar_node_type)].add(
                        (src_node_id, tar_node_id)
                    )
                    # relations[(tar_node_type, triple[1], src_node_type)].add(
                    #     (tar_node_id, src_node_id)
                    # )
            edge_index_dict = {}
            for key, edges in relations.items():
                src_node_type, relation_type, tar_node_type = key
                src_node_ids, tar_node_ids = zip(*edges)
                src_node_ids = torch.tensor(src_node_ids)
                tar_node_ids = torch.tensor(tar_node_ids)
                tmp_edge_index = torch.stack([src_node_ids, tar_node_ids], dim=0)
                sorted_indices = torch.argsort(tmp_edge_index[0, :])
                sorted_edge_index = tmp_edge_index[:, sorted_indices]
                unique_edges = torch.unique(sorted_edge_index, dim=1)
                edge_index_dict[key] = unique_edges

            ###  Initialize the data object ###
            data = HeteroData()
            data["user"].x = torch.tensor(user_embeddings)
            data["keyword"].x = torch.tensor(keyword_embeddings)
            data["tweet"].x = torch.tensor(tweet_embeddings)
            data["user"].y = torch.tensor(labels)

            for (src_type, relation_type, dst_type), edges in edge_index_dict.items():
                data[src_type, relation_type, dst_type].edge_index = edges

            torch.save(data, save_or_load_path)
            logger.info(data.metadata())
            logger.info(f"Data saved to: {save_or_load_path}")

        dataname = module_config.config.name
        data_dict[dataname.lower()] = data
        self.data.update(data_dict)

    def LoadDataLoader(self, module_config):
        option = module_config.option
        use_column = module_config.use_column
        target_node_type = self.config.train.additional.target_node_type
        save_or_load_path = osp.join(self.config.DATA_FOLDER, module_config.path)
        if osp.exists(save_or_load_path) and option == "default":
            split_masks = torch.load(save_or_load_path)
        else:
            y_true = self.data[use_column][target_node_type].y
            split_masks = create_split_masks(y_true, **module_config.split_ratio)
            torch.save(split_masks, save_or_load_path)
        for mode in module_config.config.keys():
            for data_config in module_config.config[mode]:
                use_split = data_config.split
                dataset_type = data_config.dataset_type
                data_loader = self.data[use_column]
                data_loader[target_node_type].mask = split_masks[use_split]
                loader_name = f"{mode}/{dataset_type}.{use_split}"
                self.data_loaders[mode][loader_name] = data_loader
                logger.info(
                    f"[Data Statistics]: {mode} data loader: {loader_name} {len(data_loader)}"
                )
        # for dataname in self.data.keys():
        #     if module_config.config.get("RandomNodeSplit", None):
        #         data_config = module_config.config.RandomNodeSplit
        #         SplitClass = RandomNodeSplit(**data_config)
        #         self.data[dataname] = SplitClass(self.data[dataname])
        #     for mode in ["train", "valid", "test"]:
        #         data_loader = DataLoader(
        #             [self.data[dataname]],
        #             batch_size=self.config[mode].batch_size,
        #             shuffle=True,
        #             # follow_batch=["x"],
        #         )
        #         self.data_loaders[mode][f"{mode}/{dataname}"] = data_loader
        #         logger.info(
        #             f"[Data Statistics]: {mode} data loader: {mode} {len(data_loader)}"
        #         )