import numpy as np
import torch

def build_graph(keypoints):
    num_points = keypoints.shape[0]
    
    # Define edges based on human hand anatomy
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),   # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),   # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Little finger
    ]
    
    # Create adjacency matrix
    adj_matrix = np.zeros((num_points, num_points))
    for i, j in edges:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    
    # Compute position encoding (e.g., using node degree)
    degree_matrix = np.diag(adj_matrix.sum(axis=1))
    pos_encoding = np.linalg.inv(degree_matrix + np.eye(num_points)).dot(adj_matrix)
    
    return torch.tensor(adj_matrix, dtype=torch.float32), torch.tensor(pos_encoding, dtype=torch.float32)

# Example usage:
# keypoints = np.random.rand(21, 2)  # Example keypoints
# adj_matrix, pos_encoding = build_graph(keypoints)
