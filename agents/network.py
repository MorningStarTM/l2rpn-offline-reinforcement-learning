import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

class QNetwork(nn.Module):
    def __init__(self, cfg):
        super(QNetwork, self).__init__()

        self.observation_dim = cfg['observation_dim']
        self.action_dim = cfg['action_dim']
        self.fc1_dim = cfg['FC1']
        self.fc2_dim = cfg['FC2']
        self.lr = cfg['lr']
        self.fc1 = nn.Linear(self.observation_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc1_dim)
        self.fc3 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.fc4 = nn.Linear(self.fc2_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
    



class GraphNetwork(nn.Module):
    def __init__(self, node_feature_dim, action_dim, cfg):
        super(GraphNetwork, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.action_dim = action_dim
        self.gc1_dim = cfg['GC1']
        self.fc1_dim = cfg['FC1']
        self.fc2_dim = cfg['FC2']
        self.gc1 = gnn.GCNConv(self.node_feature_dim, self.gc1_dim)
        self.gc2 = gnn.GCNConv(self.gc1_dim, self.gc1_dim)
        self.fc1 = nn.Linear(441, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc2_dim, self.fc2_dim)
        self.fc3  = nn.Linear(self.fc2_dim, self.action_dim)

    def forward(self, state, adj):
        x = F.relu(self.gc1(state, adj))
        x = F.relu(self.gc2(x, adj))
        x = x.view(x.size(0), -1)[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x