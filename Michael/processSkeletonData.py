import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import pymaid as pm
import pandas as pd
from scipy.signal import resample

mb_modelling_path = os.getenv("mb_modelling_path")
sys.path.append(mb_modelling_path)

from credentials.pm_keys import username, token, auth_pass
url = "https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/drosophila/l1/seymour/"


def getNeuronClassIndicesAndSkeletons():
    
    # Define save names
    neuron_skeletons_fname = "./data/neuron_skeletons.npy"

    # Load from saved (if they exist)
    if os.path.exists(neuron_skeletons_fname):
        neuron_skeletons = np.load(neuron_skeletons_fname, allow_pickle=True).item()
        return neuron_skeletons
    
    # Load Winding et al. 2023 neuron types
    syn_data = pd.read_csv("ad_connectivity_matrix.csv", index_col=0)
    syn_data.sort_index(inplace=True)
    skids = syn_data.index.values

    # Login to Catmaid
    pm.CatmaidInstance(server=url, api_token=token, http_user=username, http_password=auth_pass)

    # Generate neuron fragments
    neuron_skeletons = {}
    for skid in tqdm(skids):
        # Get arbour data
        neuron_arbours = pm.get_arbor([skid])
        # Get skeleton points    
        skeleton_points = [neuron_arbours.iloc[i].nodes[["x","y","z"]].values.tolist() for i in range(len(neuron_arbours))]
        neuron_skeletons[skid] = skeleton_points

    # Save class SKIDs and skeletons
    np.save("./data/neuron_skeletons.npy", neuron_skeletons)
    
    return neuron_skeletons

def getNeuronFragments(skeleton, n_fragments, fragment_max_size, fragment_radius, gridifying_factor):
    # Gridify skeleton (to remove fine detail of neural arbours). This steps helps to sample more evenly along the neuron
    skeleton_grid = np.round(skeleton / gridifying_factor) * gridifying_factor
    skeleton_grid = np.unique(skeleton_grid, axis=0)
    # Get fragments
    fragments = []
    for i in range(n_fragments):
        # Get random point
        random_idx = np.random.randint(0, len(skeleton_grid))
        random_point = skeleton_grid[random_idx]
        # Find all points within radius
        distances = np.linalg.norm(skeleton - random_point, axis=1)
        fragment = skeleton[distances < fragment_radius]
        # Resample points if size is greater than maximum size
        if fragment.shape[0] > fragment_max_size:
            fragment = resample(fragment, fragment_max_size)
        fragments.append(fragment)
    return fragments

def convertAttentionMaskToTrainingForm(attention_mask, n_examples, num_heads, fragment_max_size, device):
    full_attention_mask = torch.zeros([n_examples, num_heads, fragment_max_size, fragment_max_size], dtype=torch.bool)
    for h in range(num_heads):
        for i in range(n_examples):
            full_attention_mask[i,h,:,:] = attention_mask[i,:]
    full_attention_mask = full_attention_mask.to(device)
    return full_attention_mask

def generateTrainingData(neuron_skeletons, neuron_classes_to_take, neuron_classes_idx, n_fragments_per_neuron, fragment_max_size, n_neuron_types, min_fragment_radius, fragment_radius_var, gridifying_factor, shuffle=True):
    # Initialise training data
    training_data = np.zeros((len(neuron_classes_to_take), n_fragments_per_neuron, fragment_max_size, 3))
    target_data = np.zeros((len(neuron_classes_to_take), n_fragments_per_neuron, n_neuron_types))
    # Iterate over neuron types
    for n,n_type_idx in enumerate(neuron_classes_to_take):
        # Get current neuron skeletons
        cur_skeleton = neuron_skeletons[n_type_idx]
        # Take just one cell if there are multiple
        if len(cur_skeleton) > 1:
            # Get subsample
            cur_skeleton = cur_skeleton[np.random.randint(0, len(cur_skeleton))]
        # Convert skeleton to numpy array
        cur_skeleton = np.array(cur_skeleton)
        if cur_skeleton.shape[0] == 1:
            cur_skeleton = cur_skeleton[0]
        # Get fragments
        cur_fragment_radius = min_fragment_radius + np.random.uniform(0, fragment_radius_var)
        fragments = getNeuronFragments(cur_skeleton, n_fragments=n_fragments_per_neuron, fragment_max_size=fragment_max_size, fragment_radius=cur_fragment_radius, gridifying_factor=gridifying_factor)
        # Fill training data
        for i in range(n_fragments_per_neuron):
            training_data[n, i, -fragments[i].shape[0]:] = fragments[i]
            target_data[n, i, neuron_classes_idx[n]] = 1
    # Flatten training and target data
    training_data = training_data.reshape((len(neuron_classes_to_take)*n_fragments_per_neuron), fragment_max_size, 3)
    target_data = target_data.reshape((len(neuron_classes_to_take)*n_fragments_per_neuron), n_neuron_types)
    # Randomly shuffle data
    if shuffle==True:
        idx = np.random.permutation(len(training_data))
        training_data = training_data[idx]
        target_data = target_data[idx]
    # Generate attention mask
    attention_mask = np.array(training_data[:,:,0]==0, dtype=int)
    # Convert to torch tensors
    training_data = torch.tensor(training_data, dtype=torch.float32)
    target_data = torch.tensor(target_data, dtype=torch.float32)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float32)
    return training_data, target_data, attention_mask