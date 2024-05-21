import os
import random
import torch
from feature_engineering import extract_features
from siamese_mlp import SiameseMLP
from triplet_sampling import load_dataset, load_keypoints

def save_prototypes(prototypes, file_path='prototypes.txt'):
    with open(file_path, 'w') as f:
        for label, prototype in prototypes.items():
            prototype_str = ' '.join(map(str, prototype.tolist()))
            f.write(f'{label} {prototype_str}\n')

def load_prototypes(file_path='prototypes.txt'):
    prototypes = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = parts[0]
            prototype = torch.tensor(list(map(float, parts[1:])), dtype=torch.float32)
            prototypes[label] = prototype
    return prototypes

def few_shot_learning(few_shot_dir, model, num_shots=1):
    model.eval()
    prototypes = {}
    dataset = load_dataset(few_shot_dir)

    for label, files in dataset.items():
        support_set = random.sample(files, num_shots)
        support_features = [model(extract_features(load_keypoints(file)).float().to(device)).mean(0) for file in support_set]
        prototype = torch.mean(torch.stack(support_features), dim=0)
        prototypes[label] = prototype

    save_prototypes(prototypes)

    correct = 0
    total = 0
    for label, files in dataset.items():
        for file in files:
            query_features = model(extract_features(load_keypoints(file)).float().to(device))
            distances = {proto_label: torch.norm(query_features - proto) for proto_label, proto in prototypes.items()}
            predicted_label = min(distances, key=distances.get)
            if predicted_label == label:
                correct += 1
            total += 1
    
    accuracy = 100 * correct / total
    print(f'{num_shots}-shot Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    few_shot_dir = 'dataset/few-shot'
    num_shots = 5  # Choose from 1, 5, 10
    
    input_dim = extract_features(load_keypoints('dataset/train/0.json')).shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SiameseMLP(input_dim=input_dim, hidden_dim=128, output_dim=64).to(device)
    model.load_state_dict(torch.load('models/siamese_mlp.pth'))
    
    few_shot_learning(few_shot_dir, model, num_shots)
