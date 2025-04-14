import networkx as nx 
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_undirected




def extract_stat(train_edges_final):
    """
    Computes standard graph statistics for each node based on the positive edges in the graph.

    Inputs:
    - train_edges_final (list of tuples): list of triplets (u, v, label) from the training data,
                                          where only positive links (label == 1) are used to build the graph.

    Outputs:
    - deg_dict (dict): node degree for each node.
    - clustering_dict (dict): clustering coefficient for each node.
    - pagerank_dict (dict): PageRank score for each node (alpha=0.85).
    - closeness_dict (dict): closeness centrality for each node.
    - betweenness_dict (dict): betweenness centrality for each node.
    """

    # Build undirected NetworkX graph from positive training edges
    G = nx.Graph()
    for u, v, label in train_edges_final:
        if label == 1:
            G.add_edge(u, v)

    # Compute statistics
    deg_dict = dict(G.degree())
    clustering_dict = nx.clustering(G)
    pagerank_dict = nx.pagerank(G, alpha=0.85)
    closeness_dict = nx.closeness_centrality(G)
    betweenness_dict = nx.betweenness_centrality(G)
    
    return deg_dict, clustering_dict, pagerank_dict, closeness_dict, betweenness_dict






def compute_nodewise_stat(deg_dict, clustering_dict, pagerank_dict, closeness_dict, betweenness_dict, src, dst, device):
    """
    Constructs feature vectors for each node pair by concatenating their graph statistics.

    Inputs:
    - deg_dict (dict): dictionary of node degrees.
    - clustering_dict (dict): dictionary of clustering coefficients.
    - pagerank_dict (dict): dictionary of PageRank scores.
    - closeness_dict (dict): dictionary of closeness centralities.
    - betweenness_dict (dict): dictionary of betweenness centralities.
    - src (list or tensor): indices of source nodes in the (u, v) pairs.
    - dst (list or tensor): indices of destination nodes in the (u, v) pairs.
    - device (torch.device): device on which to return the tensor (CPU or GPU).

    Outputs:
    - out (torch.Tensor): tensor of shape [batch_size, 10] containing the concatenated statistics (5 per node).
    """

    out = []
    for u, v in zip(src, dst):
        # Node-wise features
        deg_u = deg_dict.get(u, 0)
        deg_v = deg_dict.get(v, 0)
        clu_u = clustering_dict.get(u, 0.0)
        clu_v = clustering_dict.get(v, 0.0)
        pr_u = pagerank_dict.get(u, 0.0)
        pr_v = pagerank_dict.get(v, 0.0)
        cls_u = closeness_dict.get(u, 0.0)
        cls_v = closeness_dict.get(v, 0.0)
        bet_u = betweenness_dict.get(u, 0.0)
        bet_v = betweenness_dict.get(v, 0.0)

        node_stats = [
            deg_u, deg_v,
            clu_u, clu_v,
            pr_u, pr_v,
            cls_u, cls_v,
            bet_u, bet_v
        ]
        out.append(node_stats)
    return torch.tensor(out).to(device)








def compute_node_2_vec_embed(train_edges, test_edges, device):
    """
    Trains a Node2Vec model on the co-occurrence graph and returns the learned node embeddings.

    Inputs:
    - train_edges (list of tuples): list of training node pairs (with or without labels).
    - test_edges (list of tuples): list of node pairs from the test set.
    - device (torch.device): device (CPU or GPU) used to train the model and store the embeddings.

    Outputs:
    - node2vec_embeddings (torch.Tensor): embedding matrix of shape [num_nodes, embedding_dim],
                                          containing the learned Node2Vec representations for each node.
    """

    # Union of all known pairs (ignoring labels)
    expanded_edges = set(train_edges + test_edges)
    graph_edges_nv = list(expanded_edges)
    node2vec_edge_index = torch.tensor(graph_edges_nv, dtype=torch.long).t().contiguous()
    node2vec_edge_index = to_undirected(node2vec_edge_index)

    # Train Node2Vec on the co-occurrence graph
    node2vec = Node2Vec(
        node2vec_edge_index,
        embedding_dim=64,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        sparse=True
    ).to(device)

    loader = node2vec.loader(batch_size=128, shuffle=True)
    optimizer_n2v = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

    print("Training Node2Vec...")
    for epoch in range(1, 15):  # 5 epochs is usually enough for small graphs
        total_loss = 0
        node2vec.train()
        for pos_rw, neg_rw in loader:
            optimizer_n2v.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer_n2v.step()
            total_loss += loss.item()
        print(f"Node2Vec Epoch {epoch}, Loss: {total_loss:.4f}")

    # Extract Node2Vec embeddings
    node2vec.eval()
    with torch.no_grad():
        node2vec_embeddings = node2vec.embedding.weight.data.clone()
        
    return node2vec_embeddings