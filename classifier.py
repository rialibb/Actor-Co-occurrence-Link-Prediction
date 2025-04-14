import torch.nn as nn
import torch
import torch.nn.functional as F





class LinkPredictor(nn.Module):
    """
    Link prediction classifier that takes two node embeddings and additional node statistics as input
    and outputs a score representing the likelihood of a link between the nodes.

    Inputs:
    - embed_dim (int): dimensionality of each node embedding (before concatenation).
    - hidden_dim (int): size of the hidden layer in the MLP (default = 64).

    Forward Inputs:
    - emb_u (torch.Tensor): embeddings of source nodes, shape [batch_size, embed_dim].
    - emb_v (torch.Tensor): embeddings of destination nodes, shape [batch_size, embed_dim].
    - embed_ns (torch.Tensor): graph-theoretical node statistics for each pair, shape [batch_size, 10].

    Output:
    - x (torch.Tensor): raw link prediction score (logit), shape [batch_size, 1].
    """

    def __init__(self, embed_dim, hidden_dim=64):
        super(LinkPredictor, self).__init__()
        # A simple feedforward network for link prediction
        self.fc1 = nn.Linear(embed_dim * 2 + 10, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, 1)
        
    def forward(self, emb_u, emb_v, embed_ns):
        # emb_u and emb_v are the embeddings for two nodes (of shape [batch_size, embed_dim])
        # Concatenate the embeddings
        #x = torch.cat([emb_u, emb_v, emb_u+emb_v, emb_u-emb_v], dim=1)  # shape [batch_size, 2*embed_dim]
        
        x = torch.cat([emb_u, emb_v, embed_ns], dim=1)  # shape [batch_size, 2*embed_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # output shape [batch_size, 1]
        return x  # raw score (logit) for link existence