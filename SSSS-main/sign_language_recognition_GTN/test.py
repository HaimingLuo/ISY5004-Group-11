import torch
import torch.nn as nn
from data_loader import get_data_loaders
from model import GraphTransformer

def test_model(test_loader, model, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for keypoints, labels in test_loader:
            keypoints = keypoints.float().to(device)
            labels = labels.to(device)
            
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * keypoints.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = running_loss / len(test_loader.dataset)
    accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    test_dir = 'dataset/test'
    batch_size = 32
    
    _, test_loader = get_data_loaders('', test_dir, batch_size)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GraphTransformer(num_classes=34)
    model.load_state_dict(torch.load('models/graph_transformer.pth'))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    test_model(test_loader, model, criterion)
