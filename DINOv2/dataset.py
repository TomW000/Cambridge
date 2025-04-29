from utils import os, T, h5py, np, tqdm
import torch
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    def __init__(self, 
                 data_name: str,
                 batch_size: int = 32, 
                 z_step: int = 1, 
                 z_ranges: list[int] = [0, 130],
                 transform=None):

        self.data_name = data_name
        self.batch_size = batch_size
        self.z_step = z_step 
        self.z_ranges = z_ranges
        self.samples = []
        self.class_to_idx = {}
        
        if transform:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize((224, 224)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        # Process all data during initialization
        self.process_all_data()
        
    def get_paths(self):
        """
        Get paths to all HDF5 files and their labels.
        
        Returns:
            List of tuples (file_path, label)
        """
        path = os.path.join(os.getcwd(), self.data_name) 
        all_data = []
        
        for date in os.listdir(path):
            if date == '.DS_Store':
                continue
                
            date_dir = os.path.join(path, date)
            for types in os.listdir(date_dir):
                if types == '.DS_Store':
                    continue
                    
                # Add class to class mapping if new
                if types not in self.class_to_idx:
                    self.class_to_idx[types] = len(self.class_to_idx)
                    
                types_dir = os.path.join(date_dir, types)
                for file in os.listdir(types_dir):
                    file_path = os.path.join(types_dir, file)
                    all_data.append((file_path, types))
                    
        return all_data

    def process_image(self, image, label):
        """
        Process a 3D image by extracting 2D slices along the z-axis.
        
        Args:
            image: 3D image array
            label: Class label for the image
        """
        label_idx = self.class_to_idx[label]
        z_range = range(self.z_ranges[0], self.z_ranges[1], self.z_step)
        
        for z_idx in z_range:
            # Skip if z_idx is out of bounds
            if z_idx >= image.shape[2]:
                continue
                
            # Extract 2D slice and convert to RGB by repeating channel
            img = image[:, :, z_idx]
            img_rgb = np.stack([img, img, img], axis=2)
            
            # Store the slice and its label
            self.samples.append((img_rgb, label_idx))

    def process_all_data(self):
        """
        Process all images in the dataset.
        """
        data_paths = self.get_paths()
        
        for image_path, label in tqdm(data_paths, desc='Processing Files'):
            try:
                with h5py.File(image_path, 'r') as f:
                    image = f['volumes/raw'][:]
                    self.process_image(image, label)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
                
        print(f"Processed {len(self.samples)} slices from {len(data_paths)} files")
        print(f"Classes: {self.class_to_idx}")

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (image, label) where image is the transformed image and 
                  label is the class index
        """
        img, label = self.samples[idx]
        
        # Apply transformations at access time
        if self.transform:
            img = self.transform(img)
            
        return img, label


# Example usage
if __name__ == "__main__":
    # Create dataset
    dataset = MedicalImageDataset(
        data_name="medical_images",
        batch_size=32,
        z_step=2,
        z_ranges=[0, 100]
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )
    
    # Example iteration
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: {images.shape}, {labels.shape}")
        
        # Just process a few batches for demonstration
        if batch_idx >= 2:
            break