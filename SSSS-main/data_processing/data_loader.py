import os
import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# 数据准备
def load_data(dataset_path='train'):
    data = []
    labels = []

    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(dataset_path, file_name)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                coords = [pt['x'] for pt in json_data['pts'].values()] + [pt['y'] for pt in json_data['pts'].values()]
                data.append(coords)
                labels.append(json_data['class'])

    data = np.array(data)
    labels = np.array(labels)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    return data, labels_encoded

def create_pairs(data, labels):
    pairs = []
    labels_pairs = []
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    for idx in range(len(data)):
        current_image = data[idx]
        label = labels[idx]
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(class_indices[label])
        pos_image = data[pos_idx]
        
        neg_label = random.randint(0, num_classes - 1)
        while neg_label == label:
            neg_label = random.randint(0, num_classes - 1)
        neg_idx = random.choice(class_indices[neg_label])
        neg_image = data[neg_idx]
        
        pairs += [[current_image, pos_image], [current_image, neg_image]]
        labels_pairs += [1, 0]
        
    return np.array(pairs), np.array(labels_pairs)

class HandGestureDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.pairs[idx, 0], dtype=torch.float32), \
               torch.tensor(self.pairs[idx, 1], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.float32)
