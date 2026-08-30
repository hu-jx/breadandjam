#import the validation dataset here and add it in 
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

class WildFakeDataset(Dataset):
    def __init__(self, num_samples: int, real_dir, fake_dir, transform=None):
        super().__init__()
        self.num_samples = num_samples
        self.transform = transform
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.samples = []
        
        # Map directories to labels: Real = 0, Fake = 1
        config = [
            (Path(real_dir), 0),
            (Path(fake_dir), 1)
        ]
        
        # Scan directories and pair paths with labels
        for folder_path, label in config:
            count = 0
            for p in folder_path.rglob("*"):
                if num_samples is not None and count >= num_samples:
                    break
                if p.suffix.lower() in valid_extensions:
                    self.samples.append((p, label))
                    count+=1

        self.implemented = True
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Open image and convert to RGB
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label