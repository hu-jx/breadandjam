from transformers import CLIPImageProcessor, CLIPModel 
import torch
from PIL import Image
from ..branch import Branch
# import torch.nn.functional as F
from branches.clip.attention_pool import AttentionPool

class CLIPFeatureBranch(Branch):
    def __init__(self, clip_model: CLIPModel, image_processor: CLIPImageProcessor):
        super().__init__(dim=1024,name='clip')
        self.clip_model = clip_model
        self.image_processor = image_processor
        for p in self.clip_model.parameters():
            p.requires_grad = False  # frozen feature extractor
        self.pool = AttentionPool(self.dim, num_heads=4)

    @torch.no_grad()
    def forward(self, inputs):
        vision_outputs = self.clip_model.vision_model(pixel_values=inputs)
        return vision_outputs.last_hidden_state
    
    
    def transform_data(self, img: Image.Image) -> torch.Tensor:
        res = self.image_processor(img, return_tensors = "pt")
        return res['pixel_values'].squeeze(0)