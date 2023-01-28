"""
 Copyright 2023 Bell Eapen

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import dgl
from dgl.data import DGLDataset
# import torch
import os
import numpy as np
from sklearn.neighbors import KDTree
import logging

logger = logging.getLogger(__name__)
# def create_graph(embedding, graph=None, **kwargs):
#     node_labels = torch.from_numpy(
#         embedding['name'].astype('category').cat.codes.to_numpy())
#     if graph is None:
#         graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
#     graph["embedding"] = embedding
#     return graph

def create_knn(*args):
    embeddings = []
    for arg in args:
        # All embeddings have the following keys: ['embedding', 'nodes', 'y', 'name', 'type', 'parameters']
        if ('embedding' in arg):
            logging.info(f"Creating KNN graph for {arg['name']}")
            embeddings.append(arg['embedding'])
        else:
            logging.info(f"Processing properties dict")
    embeddings = np.concatenate(embeddings, axis=1)
    # print (embeddings.shape) # 4 * (128 * 2)
    tree = KDTree(embeddings, leaf_size=2, metric='euclidean')
    print(tree)
    return tree