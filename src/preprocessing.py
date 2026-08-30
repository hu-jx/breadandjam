# from torchvision import transforms
import torch
from branch import Branch
from typing import Sequence
import random

LABELS = {
    'FAKE': 1,
    'REAL': 0
}
CLASSES = ['FAKE', 'REAL']

PIXEL_INDEX = 0
LABEL_INDEX = 1

class Preprocessing:
    def __init__(self, branches: Sequence[Branch], augment_prob=0.5):
        self.branches = branches 
        self.augment_prob = augment_prob

    def augment(self, img):
        if random.random() < self.augment_prob:
            pass  # put AlbumentationsX here
        return img

    def full_transform(self, img, train):
        """Creates a pixels dictionary for a singular image, img, with specific pixels
        for different branches since different branches need diff data transforms:
        Access output via pixels[branch_name]"""

        #takes in the img, and a list of methods to execute,  
        img = img.convert("RGB")
        #put dda here -> done for a singular image before putting through branches' trf
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

    def collate_fn(self, batch):
        """Collates pixels and labels for a batch into a singular dictionary (for separate feature branches)
        and their corresponding labels -> Fed into FusionModel for feature extraction"""
        branch_names = batch[0][0].keys() #same branch names for all
        pixel_dict = {}
        #get pixel values
        for branch in branch_names:
            pixel_dict[branch] = torch.stack([b[0][branch] for b in batch])
        labels = torch.tensor([b[LABEL_INDEX] for b in batch], dtype=torch.long)
        return pixel_dict, labels