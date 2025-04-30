from utils import *
from dataset import *
from DINOv2 import DINOv2

class Backbone(DINOv2, nn.Module):
    def __init__(self, nb_classes):
        super().__init__()
        self.nb_classes = nb_classes
        self.last_layer = list(self.backbone.modules())[-1].out_features

    def backbone(self, 
                 img_batch: list[], label):
        if img_batch:
            batch_tensor = torch.cat(img_batch, dim=0)
            batch_tensor = batch_tensor.to(self.device)
            with torch.no_grad():
                features = self(batch_tensor)
            self.latent.append(features)
            self.label_list.append(label)
            
    def MLP_Head(self,
                 x: torch.Tensor):
        self.input_dim = x.shape[1]
        assert self.input_dim == self.last_layer, f"Input dimension {self.input_dim} does not match last layer dimension {self.last_layer}"
        self.stack = nn.Sequential(
            nn.Linear(self.input_dim, 1.5*self.input_dim),
            nn.ReLU(),
            nn.Linear(1.5*self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, 0.5*self.input_dim),
            nn.ReLU(),
            nn.Linear(0.5*self.input_dim, self.nb_classes), 
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.MLP_Head(x)
        return x