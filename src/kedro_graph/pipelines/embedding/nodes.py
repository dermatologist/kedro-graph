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

TEMP_TREE = {}

def create_graph(*args):
    # ! remove this
    kdt = create_knn(*args)

    num_nodes = len(args[0]['embedding'])
    node_labels = args[0]['nodes']
    for arg in args:
        # All embeddings have the following keys: ['embedding', 'nodes', 'y', 'name', 'type', 'parameters']
        if ('embedding' in arg):
            logging.info(f"Creating graph for {arg['name']}")
            if (len(arg['embedding']) != num_nodes):
                raise ValueError("Embedding size mismatch")
            if (len(arg['nodes']) != num_nodes):
                raise ValueError("Node size mismatch")
            for idx, node_features in enumerate(arg['embedding']):
                pass
        else:
            logging.info(f"Processing properties dict")


    return kdt

def create_knn(*args):
    num_nodes = len(args[0]['embedding'])
    embeddings = []
    for arg in args:
        # All embeddings have the following keys: ['embedding', 'nodes', 'y', 'name', 'type', 'parameters']
        if ('embedding' in arg):
            logging.info(f"Creating KNN graph for {arg['name']}")
            embeddings.append(arg['embedding'])
            if (len(arg['embedding']) != num_nodes):
                raise ValueError("Embedding size mismatch")
        else:
            logging.info(f"Processing properties dict")
    embeddings = np.concatenate(embeddings, axis=1)
    # * Embedding shape is (n_samples, n_modality * n_features)
    tree = KDTree(embeddings, leaf_size=2, metric='euclidean')
    # processing weights
    Src = []
    Dst = []
    Weight = []
    for idx in range(num_nodes):
        dist, indices = tree.query(embeddings[idx:idx+1, :], k=3)
        for dist, ind in zip(dist[0], indices[0]):
            if (dist > 0):
                Src.append(idx)
                Dst.append(ind)
                Weight.append(dist)
    kdt = {
        'kd-tree': tree,
        'embeddings': embeddings,
        'Src': Src,
        'Dst': Dst,
        'Weight': Weight
    }
    return kdt