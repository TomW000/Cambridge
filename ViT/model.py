import torch 
from torch import nn
import numpy as np

device = ('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu')


class Embedding(nn.Module):
    def __init__(self, 
                 patch_size: tuple, 
                 embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.patch_length = self.patch_size[0] * self.patch_size[1]
        self.linear_projection = nn.Sequential(
            nn.Flatten(dim=1),
            nn.Linear(self.patch_length, self.embedding_dim)
            )
        
    def embedding(self, patch_list: list[torch.Tensor]):
        embeddings = []
        for patch_position, patch in enumerate(patch_list):
            embedding = torch.cat((torch.tensor([patch_position], dtype=torch.float32), self.linear_projection(patch).to(torch.float32)))
            embeddings.append(embedding)
        return embeddings
    
    
    
class MLP_Head(nn.Module):
    def __init__(self, 
                 input_dim:int, 
                 h_dims: list[int]):
        
        modules = []
        if h_dims != None:
            h_factor = [lambda x, i=i: x*i for i in np.arange(0.9, 0, -0.1)]
            h_dims = []
            for f in h_factor:
                h_dims.append(round(f(input_dim)))

        for h_dim in h_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=input_dim,
                        out_features=h_dim,
                        bias=True,
                        device=device,
                        dtype=torch.float32),
                    nn.ReLU()))
            input_dim = h_dim
        
        self.head = nn.Sequential(*modules, nn.Linear(h_dims[-1], 6), nn.Softmax(dim=0))
        self.head[-1]
        
        
        