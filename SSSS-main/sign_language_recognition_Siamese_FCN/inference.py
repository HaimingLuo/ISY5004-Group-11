import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from feature_engineering import extract_features
from siamese_mlp import SiameseMLP
from triplet_sampling import load_keypoints

def load_prototypes(file_path='prototypes.txt'):
    prototypes = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = parts[0]
            prototype = torch.tensor(list(map(float, parts[1:])), dtype=torch.float32)
            prototypes[label] = prototype
    return prototypes

def infer(model, keypoints, prototypes, threshold=0.7):
    keypoints = extract_features(keypoints)
    keypoints = torch.tensor(keypoints, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        embedding = model(keypoints).cpu().numpy().reshape(1, -1)
    
    best_label = None
    max_similarity = -1
    similarities = {}
    
    for label, prototype in prototypes.items():
        prototype = prototype.numpy().reshape(1, -1)
        similarity = cosine_similarity(embedding, prototype)[0][0]
        similarities[label] = similarity
        if similarity > max_similarity:
            max_similarity = similarity
            best_label = label
    
    if max_similarity >= threshold:
        return best_label, similarities
    else:
        return "Unknown", similarities

if __name__ == "__main__":
    input_file = 'path/to/input_keypoints.json'  # Modify this to the input file path
    model_path = 'models/siamese_mlp.pth'
    prototypes_path = 'prototypes.txt'
    threshold = 0.7  # Adjust the threshold as needed
    
    # Load model
    input_dim = 128  # This should match the feature vector dimension
    hidden_dim = 128
    output_dim = 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SiameseMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load prototypes
    prototypes = load_prototypes(prototypes_path)
    
    # Load input keypoints
    keypoints = load_keypoints(input_file)
    
    # Inference
    predicted_label, similarities = infer(model, keypoints, prototypes, threshold)
    print(f'Predicted label: {predicted_label}')
    for label, similarity in similarities.items():
        print(f'{label}: {similarity:.4f}')
