import torch
def to_device(pixel_dict: dict, labels: torch.Tensor):
    if torch.cuda.is_available():
        pixel_dict = {k: v.cuda() for k, v in pixel_dict.items()}
        labels = labels.cuda()
    return pixel_dict, labels