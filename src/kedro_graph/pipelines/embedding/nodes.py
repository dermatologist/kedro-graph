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
import tensorflow as tf

logger = logging.getLogger(__name__)

def create_graph(*args):
    # ! remove this
    kdt = create_knn(*args)

    num_nodes = len(args[0]['embedding'])
    node_labels = args[0]['nodes']
    graph = dgl.graph((kdt['Src'], kdt['Dst']), num_nodes=num_nodes)

    for arg in args:
        if ('embedding' not in arg):
            logging.info(f"Processing parameters dict")
            parameters = arg

    BACKEND = parameters.get('BACKEND', 'tensorflow')

    for arg in args:
        # All embeddings have the following keys: ['embedding', 'nodes', 'y', 'name', 'type', 'parameters']
        if ('embedding' in arg):
            logging.info(f"Creating graph for {arg['name']}")
            if (len(arg['embedding']) != num_nodes):
                raise ValueError("Embedding size mismatch")
            if (len(arg['nodes']) != num_nodes):
                raise ValueError("Node size mismatch")
            # TODO: Support for pytorch backend
            if (BACKEND != 'pytorch'):
                graph.ndata[arg['name']] = tf.convert_to_tensor(arg['embedding'])
            else:
                #graph.ndata[arg['name']] = torch.tensor(arg['embedding'])
                raise NotImplementedError("Pytorch backend not implemented")
            if (arg['y'] is not None):
                y = arg['y']


    # TODO: Support for pytorch backend
    if (BACKEND != 'pytorch'):
        graph.ndata['label'] = tf.convert_to_tensor(node_labels)
        graph.ndata['y'] = tf.convert_to_tensor(y)
        graph.edata['weight'] = tf.convert_to_tensor(kdt['Weight'])
    else:
        #graph.ndata['label'] = torch.tensor(node_labels)
        #graph.ndata['y'] = torch.tensor(y)
        #graph.edata['weight'] = torch.tensor(kdt['Weight'])
        raise NotImplementedError("Pytorch backend not implemented")

    MASK = parameters.get('MASK', False)
    TRAIN = parameters.get('TRAIN', 0.6)
    VAL = parameters.get('VAL', 0.2)
    if (MASK):
        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = num_nodes
        n_train = int(n_nodes * TRAIN)
        n_val = int(n_nodes * VAL)
        if (BACKEND == 'pytorch'):
            # train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            # val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            # test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True
            test_mask[n_train + n_val:] = True
            raise NotImplementedError("Pytorch backend not implemented")
        else:
            train_mask = tf.zeros(n_nodes, dtype=tf.bool)
            val_mask = tf.zeros(n_nodes, dtype=tf.bool)
            test_mask = tf.zeros(n_nodes, dtype=tf.bool)
            train_mask = tf.concat([tf.ones(n_train, dtype=tf.bool), tf.zeros(n_nodes - n_train, dtype=tf.bool)], axis=0)
            val_mask = tf.concat([tf.zeros(n_train, dtype=tf.bool), tf.ones(n_val, dtype=tf.bool), tf.zeros(n_nodes - n_train - n_val, dtype=tf.bool)], axis=0)
            test_mask = tf.concat([tf.zeros(n_train + n_val, dtype=tf.bool), tf.ones(n_nodes - n_train - n_val, dtype=tf.bool)], axis=0)

        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask


    return graph

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
            logging.info(f"Processing parameters dict")
            parameters = arg
    embeddings = np.concatenate(embeddings, axis=1)
    # * Embedding shape is (n_samples, n_modality * n_features)
    tree = KDTree(embeddings, leaf_size=parameters['LEAF_SIZE'], metric=parameters['METRIC'])
    # processing weights
    Src = []
    Dst = []
    Weight = []
    for idx in range(num_nodes):
        dist, indices = tree.query(embeddings[idx:idx+1, :], k=parameters['NEIGHBOURS'])
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