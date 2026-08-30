import os, io
from PIL import Image
import torch
import torchvision.transforms as T

to_tensor, to_pil = T.ToTensor(), T.ToPILImage()
_vae = None

def get_vae():
    global _vae
    if _vae is None:
        from diffusers import AutoencoderKL
        _vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        _vae.eval()
    return _vae

@torch.no_grad()
def vae_reconstruct(img):
    vae = get_vae()
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

def align_all_data(raw_path, aligned_train_path="cifake_aligned/train"):
    align_folder(f"{raw_path}/train/REAL", f"{aligned_train_path}/REAL", apply_vae=True)
    align_folder(f"{raw_path}/train/FAKE", f"{aligned_train_path}/FAKE", apply_vae=False)

if __name__ == '__main__':
    import kagglehub
    raw_path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
    align_all_data(raw_path)