import torch
import numpy as np
import torch.nn as nn
import torch.functional as F
import torch.transform as T
from PIL import Image 
import os


def data_loader():
    train_x, train_y = [], []
    test_x, test_y = [], []

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
                            train_x.append(np.array(img))
                else:
                    for image in os.listdir(coord_path):
                        if image.startswith('.'):
                            continue
                        img_path = os.path.join(coord_path, image)
                        with Image.open(img_path) as img:
                            train_y.append(np.array(img))
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
                            test_x.append(np.array(img))
                else:
                    for image in os.listdir(coord_path):
                        if image.startswith('.'):
                            continue
                        img_path = os.path.join(coord_path, image)
                        with Image.open(img_path) as img:
                            test_y.append(np.array(img))

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)



class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.FC1 = nn.Linear()
        