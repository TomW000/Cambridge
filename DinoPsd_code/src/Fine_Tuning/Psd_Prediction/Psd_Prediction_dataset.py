import torch
import numpy as np
import os
import random 
from torch.utils.data import Dataset
import torch.utils.data as utils
from tqdm.notebook import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from DinoPsd import DinoPsd_pipeline
from DinoPsd_utils import get_img_processing_f

from setup import embeddings_path, feat_dim


EMBEDDINGS = torch.load(os.path.join(embeddings_path, 'small_dataset_embs_518.pt'))
EMBEDDINGS = torch.cat(EMBEDDINGS)

print('Done loading embeddings')


REFS = torch.load(os.path.join(embeddings_path, 'small_mean_ref_518_Aug=False_k=10.pt'), weights_only=False)

print('Done loading reference embeddings')


PSD_list, REST_list = [], []

nb_best_patches = 10

for image in tqdm(EMBEDDINGS, desc='Comparing embeddings to reference'):
    flattened_image = image.reshape(-1, feat_dim)
    similarity_matrix = cosine_similarity(REFS, flattened_image)
    best_indices = np.unravel_index(similarity_matrix.ravel().argsort()[-nb_best_patches:][::-1], similarity_matrix.shape)[1]
    PSD_list.extend(flattened_image[best_indices])
    REST_list.extend(np.delete(flattened_image, best_indices, axis=0))


REST_list = random.choices(REST_list, k=len(PSD_list))

REST = torch.stack(REST_list)
PSD = torch.stack(PSD_list)

DATA = np.concatenate([REST, PSD], axis=0)

#assert int(REST.shape[0]) == int(PSD.shape[0]), 'Unbalanced dataset'


REST_LABELS = np.zeros((REST.shape[0], 2))
REST_LABELS[:,0] = 1

PSD_LABELS = np.zeros((PSD.shape[0], 2))
PSD_LABELS[:,1] = 1

LABELS = np.concatenate([REST_LABELS, PSD_LABELS], axis = 0)

#assert int(LABELS.shape[0]) == int(DATA.shape[0]), 'Problem with labelization procedure'


DATASET = list(zip(DATA, LABELS))

DATASET = random.sample(DATASET, len(DATASET))


test_proportion = 0.2


SPLIT = int(len(DATASET)*test_proportion)
TRAINING_SET = DATASET[SPLIT:]
TEST_SET = DATASET[:SPLIT]


class Custom_Detection_Dataset(Dataset):
    def __init__(self, 
                 set):
        if set == 'training':
            self.data = TRAINING_SET
        else:
            self.data = TEST_SET

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



train_batch_size, test_batch_size = 50, 50

training_dataset = Custom_Detection_Dataset('training') 
test_dataset = Custom_Detection_Dataset('test')

training_loader = utils.DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = utils.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)