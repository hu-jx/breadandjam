import kagglehub
from preprocessing import train_transform, add_label
from torchvision import datasets
from torch.utils.data import Dataset, Subset
from torch import randperm

path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")

def fetch_train_data() -> datasets.DatasetFolder:
    dataset_path = f"{path}/train"
    dataset = datasets.ImageFolder(root=dataset_path, 
                                   transform=train_transform, 
                                   target_transform= add_label)
    return dataset

def load_subset_dataset(dataset: datasets.DatasetFolder, num_samples: int) -> Dataset:
    # num_samples = 1000
    indices = randperm(n=len(dataset)).tolist()[:num_samples]
    return Subset(dataset=dataset, indices=indices)

def fetch_test_data() -> datasets.DatasetFolder:
    dataset_path = f"{path}/test"
    dataset = datasets.ImageFolder(root=dataset_path, 
                                   transform=train_transform,
                                   target_transform=add_label)
    return dataset