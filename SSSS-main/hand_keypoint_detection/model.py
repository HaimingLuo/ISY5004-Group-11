import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50KeypointModel(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50KeypointModel, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 42)  # 21 keypoints * 2 (x, y)
    
    def forward(self, x):
        return self.resnet50(x)

# Function to load YOLOv3 model
def load_yolov3_model():
    # Assuming the YOLOv3 model is stored in a file named 'YOLOv3.pth'
    model_path = 'models/YOLOv3.pth'
    yolo_model = torch.load(model_path)
    yolo_model.eval()
    return yolo_model
