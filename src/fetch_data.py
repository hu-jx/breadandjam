from collections import defaultdict
import kagglehub
from preprocessing import Preprocessing
from torchvision import datasets
from torch.utils.data import Dataset, Subset
from wildfake_dataset import WildFakeDataset

PATH = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
class DataFetch:
    def __init__(self, preprocessing: Preprocessing):
        self.preprocessing = preprocessing

    def get_subset(self, dataset, num_samples: int) -> Dataset:
        """Returns a subset of the lazy dataset, of which returned values are of random indices"""
        class_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, target = dataset[idx]
            class_indices[target].append(idx)

        selected_indices = []
        for target, indices in class_indices.items():
            chosen = indices[:num_samples]
            selected_indices.extend(chosen)
        return Subset(dataset=dataset, indices=selected_indices)
    
    def fetch_data(self, num_samples: int, train: bool = False, test: bool = False, val: bool = False) -> Dataset:
        """Fetches the data via datasets.ImageFolder and applies full_transform lazily.
        One image is transformed at a time as this is loaded into the DataLoader
        num_samples:int = Number of samples to extract.
        train:bool = If the dataset to be returned is train dataset or not"""
        if (train or test):
            if train:
                dataset_path = f"{PATH}/train"
            else:
                dataset_path = f"{PATH}/test"

            dataset = datasets.ImageFolder(root=dataset_path, 
                                    transform=self.preprocessing.full_transform, 
                                    target_transform= self.preprocessing.add_label)
            return self.get_subset(dataset, num_samples=num_samples)
        elif (val):
            dataset = WildFakeDataset(num_samples = num_samples, 
                                      real_dir= f"{PATH}/test/REAL", 
                                      fake_dir=f"{PATH}/test/FAKE", 
                                      transform=self.preprocessing.full_transform)
            if not (dataset.implemented):
                return self.fetch_data(num_samples=num_samples, test=True)
            return dataset
        
        raise Exception('Type of dataset not specified.')

        