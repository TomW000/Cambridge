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
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from math import ceil, floor



#TODO: For LINUX:
#dataset_path = '/home/tomwelch/Cambridge/Datasets/neurotransmitter_data'

#TODO: For MAC:
dataset_path = '/Users/tomw/Documents/MVA/Internship/Cambridge/Datasets/neurotransmitter_data'

dates = sorted(glob(os.path.join(dataset_path, '*')))
neurotransmitters = sorted(list(map(lambda x: os.path.basename(os.path.normpath(x)), glob(os.path.join(dates[0], '*'))))) #@param {type:"string"} 

upsample = "bilinear" #@param {type:"string", options:["bilinear", "Nearest Neighbor", "None"], value-map:{bilinear:"bilinear", "Nearest Neighbor": "nearest", None:None}}
crop_shape = (140,140,1) #@param {type:"raw"}


curated_idx = [21,22,25,28,30,42,43,44,51,67,
               600,601,615,617,618,621,623,625,635,636,
               1230,1244,1256,1262,1264,1273,1364,1376,1408,1432,
               1801,1803,1815,1823,1830,1853,1858,1865,1869,1877,
               2410,2417,2418,2435,2442,2444,2446,2453,2455,2458,
               3013,3015,3026,3029,3032,3040,3044,3050,3059,3061]


#@markdown ### Model Input Settings
#@markdown Should be multiple of model patch_size
resize_size = 140 #@param {type:"integer"} #TODO: Try other values

device = torch.device('cpu')#torch.device('mps' if torch.backends.mps.is_available()
        #else 'cuda' if torch.cuda.is_available()
        #else 'cpu')
print("Device:", device)

# select model size
model_size = 'giant' #@param {type:"string", options:["small", "base", "large", "giant"]}

model_dims = {'small': 384, 'base': 768, 'large': 1024, 'giant': 1536}
assert model_size in model_dims, f'Invalid model size: ({model_size})'
model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size[0]}14_reg')
model.to(device)
model.eval()

feat_dim = model_dims[model_size]

few_shot = DinoSim_pipeline(model, model.patch_size, device, get_img_processing_f(resize_size),
                             feat_dim, dino_image_size=resize_size )
print("Model loaded")
