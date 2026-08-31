import sys
from pathlib import Path
SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
    
import io
import numpy as np
import torch
import streamlit as st
from PIL import Image
from branches.branch import Branch
from branches.clip.clip_feature import CLIPFeatureBranch
from branches.frequency_branch import FrequencyBranch
from branches.fusion_model import FusionModel
from preprocessing.preprocessing import Preprocessing
from train import set_up_vit
from paths import CHECKPOINT_PATH

def _get_device():
    if torch.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

@st.cache_resource
def load_model():
    if not Path(CHECKPOINT_PATH).exists():
        st.error(
            "no trained model found, run `python train.py` from the src folder first, "
            "then reload this app."
        )
        st.stop()

    device = _get_device()
    clip_vit, image_processor = set_up_vit()
    freq_branch = FrequencyBranch()
    branches = [CLIPFeatureBranch(clip_vit, image_processor=image_processor), freq_branch]
    model = FusionModel(branches=branches, num_classes=1)
    preprocessing = Preprocessing(branches=branches)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    freq_branch.load_state_dict(checkpoint['freq_state'])
    model.classifier_head.load_state_dict(checkpoint['head_state'])
    calibrator = checkpoint['calibrator']
    if calibrator is None:
        raise RuntimeError('calibrator not found in checkpoint')

    model.to(device)
    model.eval()
    return model, preprocessing, device, calibrator

def predict(image_bytes: bytes):
    model, preprocessing, device, calibrator = load_model()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixels = preprocessing.full_transform(img, train=False)
    pixel_dict = {}
    for branch in model.branches:
        if not isinstance(branch, Branch):
            raise TypeError('expected instance of Branch but got ', type(branch))
        name = branch.get_name()
        pixel_dict[name] = pixels[name].unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(pixel_dict)
        raw_prob = torch.sigmoid(logit).item()

    calibrated_prob = calibrator.predict_proba(np.reshape(raw_prob, (1, -1)))[:, 1][0]
    return round(float(calibrated_prob), 3)