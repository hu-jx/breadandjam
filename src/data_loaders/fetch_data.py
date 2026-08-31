from collections import defaultdict
import os
from functools import partial
import kagglehub
from data_loaders.wildfake_load import load_wildfake_data
from data_loaders.wildfake_dataset import WildFakeDataset
from  preprocessing.preprocessing import Preprocessing
from torchvision import datasets
from torch.utils.data import Dataset, Subset
import random

path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
ALIGNED_TRAIN_PATH = "cifake_aligned/train"
AIGC_OUT_DIR = "./wildfake_subset/AIGC"
NON_AIGC_OUT_DIR = "./wildfake_subset/Non-AIGC"

def ensure_aligned_dataset():
    if os.path.isdir(f"{ALIGNED_TRAIN_PATH}/REAL") and os.path.isdir(f"{ALIGNED_TRAIN_PATH}/FAKE"):
        return
    
    from  preprocessing.align_data import align_all_data
    print("aligned dataset not found, building it now (one-time)")
    align_all_data(path, ALIGNED_TRAIN_PATH)

ensure_aligned_dataset()

PATH = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
class DataFetch:
    def __init__(self, preprocessing: Preprocessing):
        self.preprocessing = preprocessing

    def get_subset(self, dataset, num_samples: int, seed=42) -> Dataset:
        """Returns a subset of the lazy dataset, of which returned values are the first num_samples of each class"""
        class_indices = defaultdict(list)
        for idx, target in enumerate(dataset.targets):
            class_indices[target].append(idx)

        selected_indices = []
        for target, indices in class_indices.items():
            r = random.Random(seed)
            r.shuffle(indices)
            chosen = indices[:num_samples]
            selected_indices.extend(chosen)
        return Subset(dataset=dataset, indices=selected_indices)
    
    def fetch_data(self, num_samples: int, train: bool = False, test: bool = False, val: bool = False, start_n: int = 0, test_robust: bool=False) -> Dataset:
        """Fetches the data via datasets.ImageFolder and applies full_transform lazily.
        One image is transformed at a time as this is loaded into the DataLoader
        num_samples:int = Number of samples to extract.
        train:bool = If the dataset to be returned is train dataset or not"""
        if (train):
            dataset_path = ALIGNED_TRAIN_PATH 
            dataset = datasets.ImageFolder(root=dataset_path, 
                                    transform=partial(self.preprocessing.full_transform, 
                                                      train=train), 
                                    target_transform= self.preprocessing.add_label)
            return self.get_subset(dataset, num_samples=num_samples)
        elif (val or test):
            load_wildfake_data(real_dir=NON_AIGC_OUT_DIR, fake_dir=AIGC_OUT_DIR)
            dataset = WildFakeDataset(num_samples = num_samples, 
                                      real_dir= NON_AIGC_OUT_DIR, 
                                      fake_dir= AIGC_OUT_DIR, 
                                      transform= partial(self.preprocessing.full_transform, 
                                                         train=False, test_robust = test_robust),
                                        start_n=start_n)
            return dataset
           
        raise Exception('Type of dataset not specified.')

        