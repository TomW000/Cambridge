import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms as T
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile


import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch

class MyDataset(Dataset):
    def __init__(self, data_path: str, split: str):
        self.dir = Path(data_path)
        self.split = split
        self.data = []
        
        self.transform = transform = T.Compose([T.Resize(224),                      # 1. Resize first
                                                T.CenterCrop(224),                  # 2. Then crop
                                                T.ToTensor(),                       # 3. Convert to tensor (0-255 → 0-1)
                                                T.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0]==1 else x),  # 4. Handle grayscale
                                                T.Normalize([0.5]*3, [0.5]*3),  # 5. Normalize
                                                T.ConvertImageDtype(torch.float32)   # 6. Ensure float32 (redundant in this case)
                                                ])
        
        target_dir = self.dir / self.split
        for image in os.listdir(target_dir):
            if image == '.DS_Store':  
                continue
                
            img_path = target_dir / image
            try:
                with Image.open(img_path) as img:
                    self.data.append(self.transform(img))
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
                self.data.append(torch.zeros(1, 256, 256))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], 0.0 
    

class VAEDataset(LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, num_workers: int = 0):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage: Optional[str] = None):
        
        self.train_dataset = MyDataset(
            data_path=self.data_path,
            split='train'
        )
        self.val_dataset = MyDataset(
            data_path=self.data_path,
            split='test',
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
        
     
     
     
# CLEARLY OVER-ENGINEERED!!!
        """ self.train_x, self.train_y = [], []
        self.test_x, self.test_y = [], []
    
        transform = T.Compose([T.PILToTensor(),
                               T.ConvertImageDtype(torch.float32)])

        data_dir = os.path.join(os.getcwd(), 'EM_Data', 'original')

        data_sets = [f for f in os.listdir(data_dir) if not f.startswith('.')]

        for data_set in data_sets:
            dataset_path = os.path.join(data_dir, data_set)

            if data_set == 'train':
                for coordinate in os.listdir(dataset_path):
                    coord_path = os.path.join(dataset_path, coordinate)
                    if not os.path.isdir(coord_path):
                        continue

                    if coordinate == 'x':
                        for image in os.listdir(coord_path):
                            if image.startswith('.'):
                                continue
                            img_path = os.path.join(coord_path, image)
                            with Image.open(img_path) as img:
                                self.train_x.append(transform(img))
                    else:
                        for image in os.listdir(coord_path):
                            if image.startswith('.'):
                                continue
                            img_path = os.path.join(coord_path, image)
                            with Image.open(img_path) as img:
                                self.train_y.append(transform(img))
            else:
                for coordinate in os.listdir(dataset_path):
                    coord_path = os.path.join(dataset_path, coordinate)
                    if not os.path.isdir(coord_path):
                        continue

                    if coordinate == 'x':
                        for image in os.listdir(coord_path):
                            if image.startswith('.'):
                                continue
                            img_path = os.path.join(coord_path, image)
                            with Image.open(img_path) as img:
                                self.test_x.append(transform(img))
                    else:
                        for image in os.listdir(coord_path):
                            if image.startswith('.'):
                                continue
                            img_path = os.path.join(coord_path, image)
                            with Image.open(img_path) as img:
                                self.test_y.append(transform(img)) """
                                
                                

