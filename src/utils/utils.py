import torch
def to_mps(pixel_dict: dict, labels: torch.Tensor):
    if torch.mps.is_available():
        pixel_dict = {k: v.to(device='mps') for k, v in pixel_dict.items()}
        labels = labels.to(device='mps')
    return pixel_dict, labels