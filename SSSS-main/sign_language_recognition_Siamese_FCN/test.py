import torch
from feature_engineering import extract_features
from siamese_mlp import SiameseMLP
from triplet_sampling import load_keypoints

def test_model(test_triplets, model):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for anchor, positive, negative in test_triplets:
            anchor_features = extract_features(load_keypoints(anchor))
            positive_features = extract_features(load_keypoints(positive))
            negative_features = extract_features(load_keypoints(negative))
            
            anchor_output = model(anchor_features.float().to(device))
            positive_output = model(positive_features.float().to(device))
            negative_output = model(negative_features.float().to(device))
            
            pos_distance = torch.norm(anchor_output - positive_output)
            neg_distance = torch.norm(anchor_output - negative_output)
            
            if pos_distance < neg_distance:
                correct += 1
            total += 1
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    test_dir = 'dataset/test'
    num_triplets = 100
    
    dataset = load_dataset(test_dir)
    triplets = create_triplet(dataset, num_triplets)
    
    input_dim = extract_features(load_keypoints(triplets[0][0])).shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SiameseMLP(input_dim=input_dim, hidden_dim=128, output_dim=64).to(device)
    model.load_state_dict(torch.load('models/siamese_mlp.pth'))
    
    test_model(triplets, model)
