import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import get_data_loaders
from model import GraphTransformer

def train_model(train_loader, model, criterion, optimizer, num_epochs=25):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for keypoints, labels in train_loader:
            keypoints = keypoints.float().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * keypoints.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
    
    return model

if __name__ == "__main__":
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'
    batch_size = 32
    num_epochs = 25
    learning_rate = 0.001
    
    train_loader, _ = get_data_loaders(train_dir, test_dir, batch_size)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GraphTransformer(num_classes=34).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model = train_model(train_loader, model, criterion, optimizer, num_epochs)
    
    torch.save(model.state_dict(), 'models/graph_transformer.pth')
