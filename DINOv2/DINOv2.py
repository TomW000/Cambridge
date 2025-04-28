from utils import torch

class DINOv2:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                       else 'mps' if torch.backends.mps.is_available() 
                       else "cpu")
        
        print(f"Using device: {self.device}")
    
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.to(self.device)
        self.model.eval()
        
    def __call__(self, x):
        return self.model(x)