import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

class HandGestureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_files = sorted(os.listdir(data_dir))
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        
        # Read data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        keypoints = np.array(data['pts'])  # 21 points, each with (x, y)
        label = data['cls']
        
        if self.transform:
            keypoints = self.transform(keypoints)
        
        return keypoints, label

# Function to get data loaders
def get_data_loaders(train_dir, test_dir, batch_size=32):
    train_dataset = HandGestureDataset(train_dir)
    test_dataset = HandGestureDataset(test_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader
