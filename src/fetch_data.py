from functools import partial
import kagglehub
from preprocessing import Preprocessing
from torchvision import datasets
from torch.utils.data import Dataset, Subset
from torch import randperm

path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
ALIGNED_TRAIN_PATH = "cifake_aligned/train"

class DataFetch:
    def __init__(self, preprocessing: Preprocessing):
        self.preprocessing = preprocessing

    def get_subset(self, dataset: datasets.DatasetFolder, num_samples: int) -> Dataset:
        """Returns a subset of the lazy dataset, of which returned values are of random indices"""
        indices = randperm(n=len(dataset)).tolist()[:num_samples]
        return Subset(dataset=dataset, indices=indices)
    
    def fetch_data(self, num_samples: int, train: bool) -> Dataset:
        """Fetches the data via datasets.ImageFolder and applies full_transform lazily.
        One image is transformed at a time as this is loaded into the DataLoader
        num_samples:int = Number of samples to extract.
        train:bool = If the dataset to be returned is train dataset or not"""
        if (train):
            dataset_path = ALIGNED_TRAIN_PATH if train else f"{path}/test"
        else:
            dataset_path = f"{path}/test"
        dataset = datasets.ImageFolder(root=dataset_path, 
                                    transform=partial(self.preprocessing.full_transform, 
                                                      train=train), 
                                    target_transform= self.preprocessing.add_label)
        return self.get_subset(dataset, num_samples=num_samples)