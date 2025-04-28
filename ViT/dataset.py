from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as T
from pathlib import Path
import h5py
import os
from torchvision.transforms import v2

class MyDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 split: str,
                 dataset_proportions: list[str]):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.dataset_proportions = dataset_proportions

        all_data = []
        for date in os.listdir(self.data_path):
            for types in os.listdir(os.path.join(self.data_path, date)):
                for file in os.listdir(os.path.join(self.data_path, date, types)):
                    all_data.append(file)

        total_length = len(all_data)
        train_split = int(dataset_proportions[0]*total_length)
        val_split = train_split + int(dataset_proportions[1]*total_length)

        if split == 'train':
            self.data = all_data[:train_split]

        elif split == 'val':
            self.data = all_data[train_split:val_split]

        elif split == 'test':
            self.data = all_data[val_split:]

        else: 
            raise ValueError(f'Unknown split: {self.split}')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        with h5py.File(self.data[idx], 'r') as f:
            
#TODO                                                                                                                                               

            
            self.transform = T.Compose([T.ToTensor(),
                                        ])
            return self.transform(f)


class ViTDataset():
    def __init__(self,
                 data_path: str, 
                 dataset_proportions: list[int],
                 batch_sizes: list[str],
                 nb_workers: list[int],
                 ):
        super().__init__()
        self.data_path = data_path
        self.dataset_proportions = dataset_proportions
        self.train_batch_size, self.val_batch_size, self.test_batch_size = batch_sizes[0], batch_sizes[1], batch_sizes[2]
        self.train_nb_workers, self.val_nb_workers, self.test_nb_workers = nb_workers[0], nb_workers[1], nb_workers[2]
        
    def setup(self):
        self.train_dataloader = MyDataset(
            data_path=self.data_path,
            split='train', 
            dataset_proportions = self.dataset_proportions
        )
        
        self.val_dataloader = MyDataset(
            data_path=self.data_path,
            split='val',
            dataset_proportions = self.dataset_proportions
        )
        
        self.test_dataloader = MyDataset(
            data_path=self.data_path,
            split='test',
            dataset_proportions = self.dataset_proportions
        )
        
    def train_dataset(self):
        return DataLoader(
            self.train_dataloader,
            num_workers=self.train_nb_workers, 
            batch_size=self.train_batch_size,
            shuffle=True)
        
    def val_dataset(self):
        return DataLoader(
            self.val_dataloader,
            num_workers = self.val_nb_workers, 
            batch_size=self.val_batch_size,
            shuffle=True)
        
    def test_dataset(self):
        return DataLoader(
            self.test_dataloader,
            num_workers=self.test_nb_workers, 
            batch_size=self.test_batch_size,
            shuffle=True)