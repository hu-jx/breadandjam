# from torchvision import transforms
import torch
from branch import Branch
from typing import Sequence
import random
import albumentations as A
import numpy as np
from PIL import Image

LABELS = {
    'FAKE': 1,
    'REAL': 0
}
CLASSES = ['FAKE', 'REAL']

PIXEL_INDEX = 0
LABEL_INDEX = 1

class Preprocessing:
    def __init__(self, branches: Sequence[Branch], augment_prob=0.5, transform_weights=None):
        self.branches = branches 
        self.augment_prob = augment_prob
        self.crop_fraction = 0.8
        self.transformations = {
            'jpeg': A.ImageCompression(quality_range=(30, 90), p=1.0),
            'blur': A.GaussianBlur(sigma_limit=(0.5, 2.0), p=1.0),
            'downscale': A.Downscale(scale_range=(0.25, 0.5), p=1.0),
            'noise': A.GaussNoise(std_range=(0.02, 0.10), p=1.0),
            'color_jitter': A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=1.0),
        }
        self.transform_names = list(self.transformations.keys()) + ['crop']
        self.transform_weights = transform_weights or {name: 1.0 for name in self.transform_names}
        self.transform_counts = {name: 0 for name in self.transform_names}

    def center_crop_fraction(self, arr): 
        # for cropping to 0.8
        h, w = arr.shape[:2]
        new_h = max(1, round(h * self.crop_fraction))
        new_w = max(1, round(w * self.crop_fraction))
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return arr[top:top + new_h, left:left + new_w]

    def augment(self, img):
        if random.random() < self.augment_prob:
            # (done) put AlbumentationsX here
            arr = np.array(img)
            names = self.transform_names
            weights = [self.transform_weights[name] for name in names]
            chosen = random.choices(names, weights=weights, k=1)[0]
            self.transform_counts[chosen] += 1
            if chosen == 'crop':
                arr = self.center_crop_fraction(arr)

            else:
                arr = self.transformations[chosen](image=arr)['image']
            return Image.fromarray(arr)
        
        return img

    def full_transform(self, img, train):
        """Creates a pixels dictionary for a singular image, img, with specific pixels
        for different branches since different branches need diff data transforms:
        Access output via pixels[branch_name]"""

        #takes in the img, and a list of methods to execute,  
        img = img.convert("RGB")
        # (done!) put dda here -> done for a singular image before putting through branches' trf 
        if train:
            img = self.augment(img)
        pixels = {}
        for branch in self.branches:
            name = branch.get_name()
            pixels[name] = branch.transform_data(img=img)
        return pixels

    def add_label(self, class_index):
        class_name = CLASSES[class_index]
        return LABELS[class_name]

    @staticmethod
    def collate_fn(batch):
        """Collates pixels and labels for a batch into a singular dictionary (for separate feature branches)
        and their corresponding labels -> Fed into FusionModel for feature extraction"""
        branch_names = batch[0][0].keys() #same branch names for all
        pixel_dict = {}
        #get pixel values
        for branch in branch_names:
            pixel_dict[branch] = torch.stack([b[0][branch] for b in batch])
        labels = torch.tensor([b[LABEL_INDEX] for b in batch], dtype=torch.float32)
        return pixel_dict, labels