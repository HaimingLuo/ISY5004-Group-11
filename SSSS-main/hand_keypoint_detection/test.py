import torch
import torch.nn as nn
from data_loader import get_data_loaders
from model import ResNet50KeypointModel

def test_model(test_loader, model, criterion):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, keypoints in test_loader:
            images = images.to(device)
            keypoints = keypoints.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            
            running_loss += loss.item() * images.size(0)
    
    test_loss = running_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

if __name__ == "__main__":
    test_image_dir = 'dataset/image/test'
    test_annotation_dir = 'dataset/groundtruth/test'
    batch_size = 32
    
    _, test_loader = get_data_loaders('', '', test_image_dir, test_annotation_dir, batch_size)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet50KeypointModel(pretrained=False)
    model.load_state_dict(torch.load('models/proposed_ResNet50_pretrained.pth'))
    model = model.to(device)
    criterion = nn.MSELoss()
    
    test_model(test_loader, model, criterion)
