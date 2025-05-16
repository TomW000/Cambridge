import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch

from napari_dinosim.dinoSim_pipeline import *
from napari_dinosim.utils import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

import h5py
from torch.nn import functional as F
import torchvision.transforms.v2.functional as T
from torchvision import transforms
from tqdm import tqdm
import matplotlib.patches as patches
from torch import nn
import torchvision.transforms as Trans
from sklearn.decomposition import PCA
import sklearn.neighbors
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import pandas as pd
import seaborn as sns
from collections import Counter
from typing import Union



#TODO: For LINUX:
dataset_path = '/home/tomwelch/Cambridge/Datasets/neurotransmitter_data'

#TODO: For MAC:
#dataset_path = '/Users/tomw/Documents/MVA/Internship/Cambridge/Datasets/neurotransmitter_data'

dates = glob(os.path.join(dataset_path, '*'))
neurotransmitters = list(map(lambda x: os.path.basename(os.path.normpath(x)), glob(os.path.join(dates[0], '*')))) #@param {type:"string"} 

upsample = "bilinear" #@param {type:"string", options:["bilinear", "Nearest Neighbor", "None"], value-map:{bilinear:"bilinear", "Nearest Neighbor": "nearest", None:None}}
crop_shape = (512,512,1) #@param {type:"raw"}

#@markdown ### Model Input Settings
#@markdown Should be multiple of model patch_size
resize_size = 518 #@param {type:"integer"}

device = torch.device('mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu')
print("Device:", device)

# select model size
model_size = 'small' #@param {type:"string", options:["small", "base", "large", "giant"]}

model_dims = {'small': 384, 'base': 768, 'large': 1024, 'giant': 1536}
assert model_size in model_dims, f'Invalid model size: ({model_size})'
model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size[0]}14_reg')
model.to(device)
model.eval()

feat_dim = model_dims[model_size]

few_shot = DinoSim_pipeline(model, model.patch_size, device, get_img_processing_f(resize_size),
                             feat_dim, dino_image_size=resize_size )
print("Model loaded")
