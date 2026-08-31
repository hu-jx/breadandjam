import torch
from utils.utils import to_mps
class CachedFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, cache_path):
        data = torch.load(cache_path)
        self.labels = data.pop('labels')
        self.features = data  # dict[branch_name] -> tensor, e.g. {'clip': ..., 'frequency': ...}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {name: feats[idx] for name, feats in self.features.items()}, self.labels[idx]

def cached_collate_fn(batch):
    device = 'mps' if torch.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu') 
    branch_names = batch[0][0].keys()
    pixel_dict = {name: torch.stack([b[0][name] for b in batch]) for name in branch_names}
    labels = torch.stack([b[1] for b in batch])
    return to_mps(pixel_dict, labels)