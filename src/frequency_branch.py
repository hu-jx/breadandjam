import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms 
from PIL import Image
from branch import Branch 

class FrequencyBranch(Branch):
    def __init__(self, output_dim=256, target_size=(224, 224)):
        super().__init__(dim=output_dim, name="frequency")

        #preprocessing and resizing of images
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.56, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        #feature extraction 
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, output_dim)
        )

    @torch.no_grad()
    def forward(self, inputs):

        #apply fourier transform
        fft_map = torch.fft.rfft2(inputs, norm="forward")

        #extract amplitude spectrum and apply stabilising log transform
        amplitude = torch.abs(fft_map)
        log_amplitude = torch.log(amplitude + 1e-6)

        #extract hidden features
        raw_features = self.feature_extractor(log_amplitude)

        #apply l2 normalisation
        normalized_features = F.normalize(raw_features, p=2, dim=1)

        return normalized_features

    def transform_data(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)

    

