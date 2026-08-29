from typing import Sequence
import torch 
import torch.nn as nn
from branch import Branch

class FusionModel(nn.Module):
    def __init__(self, branches: Sequence[Branch], num_classes):
        super().__init__()
        num_dimensions = sum([branch.get_dim() for branch in branches])
        self.classifier_head = nn.Linear(num_dimensions, num_classes)
        self.branches = nn.ModuleList(branches)

    def forward(self, pixel_dict):
        """Trains every branch with the corresponding pixel values, first extracting the feature
        via the feature branches, before concatenating everything together.
        Concatenated vector is used to train classifier_head"""
        feats = []
        for branch in self.branches:
            if not (isinstance(branch, Branch)):
                raise ValueError('Incompatible type in branch')
            px = branch(pixel_dict[branch.get_name()])
            feats.append(px)
        combined = torch.cat(feats, dim=1)
        return self.classifier_head(combined)