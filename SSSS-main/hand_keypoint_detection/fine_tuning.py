import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import get_data_loaders
from model import ResNet50KeypointModel

def finetune_model(train_loader, model, criterion, optimizer, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for images, keypoints in train_loader:
            images = images.to(device)
            keypoints = keypoints.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
    
    return model

if __name__ == "__main__":
    train_image_dir = 'dataset/image/train'
    train_annotation_dir = 'dataset/groundtruth/train'
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    
    train_loader, _ = get_data_loaders(train_image_dir, train_annotation_dir, '', '', batch_size)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet50KeypointModel(pretrained=False)
    model.load_state_dict(torch.load('models/proposed_ResNet50_pretrained.pth'))
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model = finetune_model(train_loader, model, criterion, optimizer, num_epochs)
    
    torch.save(model.state_dict(), 'models/proposed_ResNet50_finetuned.pth')
