import torch
import numpy as np
import os
from random import sample
from torch.utils.data import Dataset
import torch.utils.data as utils

from DinoPsd import DinoPsd_pipeline
from DinoPsd_utils import get_img_processing_f
from Fine_Tuning.compute_embeddings import compute_embeddings

from setup import resize_size, embeddings_path, neurotransmitters, feat_dim


DATA = torch.load(os.path.join(embeddings_path, 'small_dataset_embs_518.pt'))
print('Done loading embeddings')



LABELS = np.hstack([[neuro]*int((resize_size/14)**2 * 600) for neuro in neurotransmitters]).reshape(-1, 1)

DATA = torch.cat(DATA)

DATA = DATA.reshape(-1, feat_dim)

DATASET = list(zip(DATA, LABELS))

DATASET = sample(DATASET, len(DATASET))


test_proportion = 0.2


SPLIT = int(len(DATASET)*test_proportion)
TRAINING_SET = DATASET[SPLIT:]
TEST_SET = DATASET[:SPLIT]

one_hot_neurotransmitters = np.eye(len(neurotransmitters))


class Custom_LP_Dataset(Dataset):
    def __init__(self, 
                 set):
        if set == 'training':
            self.data = TRAINING_SET
        else:
            self.data = TEST_SET

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding, label = self.data[idx]
        label_idx = neurotransmitters.index(label[0])
        return embedding, one_hot_neurotransmitters[label_idx]
    
    
train_batch_size, test_batch_size = 50, 50

training_dataset = Custom_LP_Dataset('training') 
test_dataset = Custom_LP_Dataset('test')

training_loader = utils.DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
test_loader = utils.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, pin_memory=True)