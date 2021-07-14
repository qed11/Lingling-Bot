import torch
import os
from torchvision.datasets.folder import default_loader

class AutoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform = None, loader = default_loader) -> None:
        self.root = root
        self.data = []
        for file in os.listdir(self.root):
            self.data.append(file)
        self.loader = loader
        self.transform = transform
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path = self.data[idx] #get the path of the single data
        image = self.loader(path) #convert that into an image
        if self.transform:
            image = self.transform(image) #if transform, transform
        return image