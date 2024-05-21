import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from graph_builder import build_graph

class GraphTransformer(nn.Module):
    def __init__(self, num_classes=34):
        super(GraphTransformer, self).__init__()
        self.num_points = 21
        self.d_model = 128
        self.num_layers = 3
        
        self.pos_encoder = nn.Parameter(torch.zeros(1, self.num_points, self.d_model))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8), 
            num_layers=self.num_layers
        )
        self.fc1 = nn.Linear(self.num_points * self.d_model, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.pos_encoder.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, keypoints):
        # keypoints shape: (batch_size, num_points, 2)
        batch_size = keypoints.size(0)
        
        adj_matrix, pos_encoding = build_graph(keypoints)
        
        # Encode positions
        x = torch.cat([keypoints, pos_encoding.repeat(batch_size, 1, 1)], dim=2)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Flatten and pass through FC layers
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# Example usage:
# model = GraphTransformer(num_classes=34)
