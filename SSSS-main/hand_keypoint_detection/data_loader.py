import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HandKeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.annotation_files = sorted(os.listdir(annotation_dir))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        annot_path = os.path.join(self.annotation_dir, self.annotation_files[idx])
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read annotations
        with open(annot_path, 'r') as f:
            annotations = json.load(f)
        
        # Extract keypoints
        keypoints = np.array(annotations['pts']).flatten()
        
        if self.transform:
            image = self.transform(image)
        
        return image, keypoints

# Function to get data loaders
def get_data_loaders(train_image_dir, train_annotation_dir, test_image_dir, test_annotation_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = HandKeypointDataset(train_image_dir, train_annotation_dir, transform=transform)
    test_dataset = HandKeypointDataset(test_image_dir, test_annotation_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader
