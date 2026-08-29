#interface for branches 
import torch.nn as nn
from abc import abstractmethod
from torch import Tensor
from PIL import Image

class Branch(nn.Module):
    def __init__(self, dim, name):
        super().__init__()
        self.dim = dim
        self.branch_name = name

    def get_name(self) -> str:
        return self.branch_name
    
    def get_dim(self) -> int:
        return self.dim
    
    @abstractmethod
    def transform_data(self, img: Image.Image) -> Tensor:
        """Transform data abstract method"""
