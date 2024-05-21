import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from feature_engineering import extract_features
from siamese_mlp import SiameseMLP, contrastive_loss
from triplet_sampling import load_dataset, create_triplet, load_keypoints
from data_loader import get_data_loaders

class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        anchor_features = extract_features(load_keypoints(anchor))
        positive_features = extract_features(load_keypoints(positive))
        negative_features = extract_features(load_keypoints(negative))
        return anchor_features, positive_features, negative_features

def save_prototypes(prototypes, file_path='prototypes.txt'):
    with open(file_path, 'w') as f:
        for label, prototype in prototypes.items():
            prototype_str = ' '.join(map(str, prototype.tolist()))
            f.write(f'{label} {prototype_str}\n')

def train_model(train_loader, model, criterion, optimizer, num_epochs=25):
    model.train()
    prototypes = {}
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.float().to(device), positive.float().to(device), negative.float().to(device)
            
            optimizer.zero_grad()
            
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)
            
            loss = criterion(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * anchor.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
    
    # Calculate prototype vectors
    with torch.no_grad():
        for anchor, _, label in train_loader.dataset:
            anchor_features = extract_features(load_keypoints(anchor))
            anchor_output = model(anchor_features.float().to(device)).detach().cpu()
            if label not in prototypes:
                prototypes[label] = []
            prototypes[label].append(anchor_output)
    
    for label in prototypes:
        prototypes[label] = torch.mean(torch.stack(prototypes[label]), dim=0)
    
    save_prototypes(prototypes)

if __name__ == "__main__":
    train_dir = 'dataset/train'
    batch_size = 32
    num_epochs = 25
    learning_rate = 0.001
    num_triplets = 1000
    
    train_loader = get_data_loaders(train_dir, batch_size)
    
    input_dim = extract_features(load_keypoints(train_loader.dataset[0][0])).shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SiameseMLP(input_dim=input_dim, hidden_dim=128, output_dim=64).to(device)
    criterion = contrastive_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_model(train_loader, model, criterion, optimizer, num_epochs)
    
    torch.save(model.state_dict(), 'models/siamese_mlp.pth')
