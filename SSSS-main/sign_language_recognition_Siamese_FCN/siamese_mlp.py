import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SiameseMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def contrastive_loss(anchor, positive, negative, margin=1.0):
    pos_distance = F.pairwise_distance(anchor, positive)
    neg_distance = F.pairwise_distance(anchor, negative)
    loss = torch.mean(F.relu(pos_distance - neg_distance + margin))
    return loss

# Example usage:
# model = SiameseMLP(input_dim=50, hidden_dim=128, output_dim=64)

