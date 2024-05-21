import os
import random
import json
import numpy as np
from feature_engineering import extract_features

def load_dataset(data_dir):
    data = {}
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        data[label] = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
    return data

def create_triplet(dataset, num_triplets):
    triplets = []
    labels = list(dataset.keys())
    for _ in range(num_triplets):
        anchor_label = random.choice(labels)
        positive_label = anchor_label
        negative_label = random.choice([label for label in labels if label != anchor_label])
        
        anchor = random.choice(dataset[anchor_label])
        positive = random.choice(dataset[positive_label])
        negative = random.choice(dataset[negative_label])
        
        triplets.append((anchor, positive, negative))
    
    return triplets

def load_keypoints(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data['pts']).flatten()


dataset = load_dataset('dataset/train')
triplets = create_triplet(dataset, num_triplets=100)
anchor, positive, negative = load_keypoints(triplets[0][0]), load_keypoints(triplets[0][1]), load_keypoints(triplets[0][2])
