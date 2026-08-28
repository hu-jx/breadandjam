# from torchvision import transforms
import torch
from torchvision.models import ResNet50_Weights
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, Resize, ToTensor

LABELS = {
    'FAKE': 1,
    'REAL': 0
}
CLASSES = ['FAKE', 'REAL']

pixel_idx = 0
label_idx = 1

#RGB Extraction 
def train_transform(img):
    #into pixel values
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms() #resize, normalize for resnet
    image = img.convert("RGB")
    return preprocess(image)

def add_label(class_index):
    class_name = CLASSES[class_index]
    return LABELS[class_name]

def collate_fn(batch):
    pixel_values = torch.stack([b[pixel_idx] for b in batch])
    labels = torch.tensor([b[label_idx] for b in batch], dtype=torch.long)
    return pixel_values, labels