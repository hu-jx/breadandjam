from transformers import CLIPImageProcessor, CLIPModel 
import torch
from PIL import Image
from branch import Branch

class CLIPFeatureBranch(Branch):
    def __init__(self, clip_model: CLIPModel, image_processor: CLIPImageProcessor):
        super().__init__(dim=768,name='clip')
        self.clip_model = clip_model
        self.image_processor = image_processor
        for p in self.clip_model.parameters():
            p.requires_grad = False  # frozen feature extractor

    @torch.no_grad()
    def forward(self, inputs):
        features = self.clip_model.get_image_features(pixel_values=inputs)
        if (isinstance(features, tuple)):
            raise TypeError('expected BaseModelOutputWithPooling, but got tuple')
        return features.pooler_output
    
    def transform_data(self, img: Image.Image) -> torch.Tensor:
        res = self.image_processor(img, return_tensors = "pt")
        return res['pixel_values'].squeeze(0)