import os, io
from PIL import Image
import torch
from diffusers import AutoencoderKL
import torchvision.transforms as T
import kagglehub

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
vae.eval()
to_tensor, to_pil = T.ToTensor(), T.ToPILImage()

@torch.no_grad()
def vae_reconstruct(img):
    x = to_tensor(img).unsqueeze(0) * 2 - 1
    latent = vae.encode(x).latent_dist.mode()
    recon = (vae.decode(latent).sample.clamp(-1, 1) + 1) / 2
    return to_pil(recon.squeeze(0))

def freq_align(img, quality=90):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def align_folder(src_dir, dst_dir, apply_vae):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
        if apply_vae:
            img = vae_reconstruct(img)
        img = freq_align(img)
        img.save(os.path.join(dst_dir, fname))

if __name__ == '__main__':
    raw_path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
    align_folder(f"{raw_path}/train/REAL", "cifake_aligned/train/REAL", apply_vae=True)
    align_folder(f"{raw_path}/train/FAKE", "cifake_aligned/train/FAKE", apply_vae=False)