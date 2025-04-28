import torch
from torchvision.transforms import v2 
from torch import Tensor


class Augmentation(v2):
    def __init__(self, 
                 batch: list[Tensor], 
                 alpha: int, 
                 m: int,
                 l: int
                 ):
        self.batch = batch
        self.alpha = alpha 
        self.m = m
        self.l = l
    def transforms(self):
        
        self.mixup = v2.MixUp(self.batch, alpha=self.alpha)
        self.mixup_list = [self.mixup[i] for i in self.mixup.keys()]
        
        self.randaug_list = []
        for i in len(self.batch):
            self.randaug = v2.RandAugment(self.batch[i], magnitude=self.m, num_ops=self.l)
        return torch.cat((self.batch, self.mixup_list, self.randaug_list))