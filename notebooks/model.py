import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import LayerNorm  # <--- thêm dòng này

# Mô hình GAT
class GAT(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, heads=8):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_features, hidden_dim, heads=heads, dropout=0.2)
        self.ln1 = LayerNorm(hidden_dim * heads)

        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=4, dropout=0.2)
        self.ln2 = LayerNorm(hidden_dim * 4)

        self.gat3 = GATConv(hidden_dim * 4, out_features, heads=1, concat=False, dropout=0.2)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.ln1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = self.ln2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat3(x, edge_index)
        return F.log_softmax(x, dim=1)