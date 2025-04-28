from utils import *
from dataset import *
from DINOv2 import DINOv2

class Model(DINOv2):
    def __init__(self):
        super().__init__()
        self.latent = []
        self.batch_size = 36
        self.z_step = 1
        self.z_ranges = [20, 21]
        self.label_list = []
        self.device = device

        print(f"Processing {len(all_data)} files...")
        self.process_all_data()
        print('Done')

    def process_all_data(self):
        for image_path, label in tqdm(all_data, desc='Files', leave=False):
            try:
                with h5py.File(image_path, 'r') as f:
                    image = f['volumes/raw'][:]
                    self.process_image(image, label)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue

    def process_image(self, image, label):
        z_range = range(self.z_ranges[0], self.z_ranges[1], self.z_step)
        img_batch = []

        for j, z_idx in enumerate(z_range):
            img = image[:, :, z_idx]
            img_rgb = np.stack([img, img, img], axis=2)
            img_tensor = transform(img_rgb).unsqueeze(0)
            img_batch.append(img_tensor)

            if len(img_batch) == self.batch_size or j == len(z_range) - 1:
                self.process_batch(img_batch, label)
                img_batch = []

    def process_batch(self, img_batch, label):
        if img_batch:
            batch_tensor = torch.cat(img_batch, dim=0)
            batch_tensor = batch_tensor.to(self.device)
            with torch.no_grad():
                features = self(batch_tensor)
            self.latent.append(features)
            self.label_list.append(label)
