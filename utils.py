import pandas as pd
import torch
import numpy as np
import random



def extract_features(NODE_INFO):
    """
    Loads node descriptors from a CSV file and prepares the feature matrix for training.

    Inputs:
    - NODE_INFO (str): path to the CSV file containing node information.
                       The first column contains node IDs, and the remaining columns are node features.

    Outputs:
    - X (torch.Tensor): feature matrix of shape [num_nodes, feature_dim], converted to a PyTorch tensor.
    - id_to_index (dict): dictionary mapping original node IDs to consecutive indices (0 to N-1).
    - feature_dim (int): dimensionality of the feature vectors (number of feature columns).
    """
    # Load node information (features) from CSV
    node_info = pd.read_csv(NODE_INFO, header=None)
    # The first column is node ID, the rest 932 columns are features
    node_ids = node_info[0].tolist()
    features = node_info.iloc[:, 1:].to_numpy(dtype=np.float32)

    # Map original node IDs to contiguous indices (0...N-1)
    id_to_index = {nid: idx for idx, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)
    feature_dim = features.shape[1]  # should be 932

    # Convert features to torch tensor
    X = torch.from_numpy(features)
    
    return X, id_to_index, feature_dim





def load_data(TRAIN_FILE, TEST_FILE, id_to_index):
    """
    Loads node pairs for training and testing, and converts their original node IDs to internal indices.

    Inputs:
    - TRAIN_FILE (str): path to the file containing labeled node pairs (label 0 or 1).
    - TEST_FILE (str): path to the file containing node pairs to be predicted (no labels).
    - id_to_index (dict): dictionary mapping original node IDs to consecutive internal indices.

    Outputs:
    - (train_edges, train_labels): 
        - train_edges (list of tuples): list of training node pairs as index tuples (u, v).
        - train_labels (list of int): list of labels for each pair (1 = link, 0 = no link).
    - test_edges (list of tuples): list of test node pairs as index tuples (u, v) with no label.
    """

    train_edges = []
    train_labels = []
    with open(TRAIN_FILE, "r") as f:
        for line in f:
            u, v, label = line.strip().split()
            u, v, label = int(u), int(v), int(label)
            train_edges.append((id_to_index[u], id_to_index[v]))
            train_labels.append(label)
    # Load test edges (pairs for which we need predictions)
    test_edges = []
    with open(TEST_FILE, "r") as f:
        for line in f:
            u, v = line.strip().split()
            u, v = int(u), int(v)
            test_edges.append((id_to_index[u], id_to_index[v]))
            
    return (train_edges, train_labels), test_edges






def split_train_val(train_edges, train_labels, val_ratio=0.1):
    """
    Performs a stratified split of the training data into training and validation sets.

    Inputs:
    - train_edges (list of tuples): list of training node pairs (u, v).
    - train_labels (list of int): corresponding labels for each pair (1 for link, 0 otherwise).
    - val_ratio (float): proportion of data to allocate to the validation set (default is 0.1).

    Outputs:
    - train_edges_final (list of tuples): training set containing triplets (u, v, label).
    - val_edges_final (list of tuples): validation set containing triplets (u, v, label).
    - train_pos (list of tuples): remaining positive pairs used to construct the graph structure.
    """

    # We will stratify the split to maintain balanced classes in each set.
    train_pos = [edge for edge, lbl in zip(train_edges, train_labels) if lbl == 1]
    train_neg = [edge for edge, lbl in zip(train_edges, train_labels) if lbl == 0]
    # Shuffle and split
    random.shuffle(train_pos)
    random.shuffle(train_neg)
    val_size_pos = int(len(train_pos) * val_ratio)
    val_size_neg = int(len(train_neg) * val_ratio)
    val_pos = train_pos[:val_size_pos]
    val_neg = train_neg[:val_size_neg]
    train_pos = train_pos[val_size_pos:]
    train_neg = train_neg[val_size_neg:]
    # Combine pos and neg for final train and val sets
    train_edges_final = [(u, v, 1) for (u, v) in train_pos] + [(u, v, 0) for (u, v) in train_neg]
    val_edges_final   = [(u, v, 1) for (u, v) in val_pos]   + [(u, v, 0) for (u, v) in val_neg]
    
    return train_edges_final, val_edges_final, train_pos





def compute_graph_edges(train_pos):
    """
    Builds the adjacency matrix in PyTorch Geometric format using the positive edges from the graph.

    Inputs:
    - train_pos (list of tuples): list of positive node pairs (u, v) from the training set.

    Outputs:
    - edge_index (torch.Tensor): a tensor of shape [2, num_edges] representing the edges of the co-occurrence graph,
                                 with each edge added in both directions to model an undirected graph.
    """

    graph_edges = train_pos  # use only positive edges for the graph structure
    # Since the co-occurrence graph is undirected, add both directions for each edge
    undirected_edges = []   # for NN
    for (u_idx, v_idx) in graph_edges:
        undirected_edges.append((u_idx, v_idx))
        undirected_edges.append((v_idx, u_idx))
    # Convert edge list to PyTorch tensor (shape [2, E])
    edge_index = torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()
    
    return edge_index