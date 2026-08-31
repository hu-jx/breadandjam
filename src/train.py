#python code for backend / model training in this folder
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from  data_loaders.cached_dataset import CachedFeatureDataset, cached_collate_fn
from  branches.clip.clip_feature import CLIPFeatureBranch
from branches.branch import Branch
from utils.utils import to_mps
from branches.fusion_model import FusionModel
from  data_loaders.fetch_data import DataFetch
from  preprocessing.preprocessing import Preprocessing
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor
from branches.frequency_branch import FrequencyBranch
import time
from eval import Inference
from paths import CHECKPOINT_PATH, TRAIN_FEATS_PATH, VAL_FEATS_PATH, TEST_FEATS_PATH, TEST_ROBUST_FEATS_PATH

NUM_WORKERS = 0
BATCH_SIZE = 64
NUM_TRAIN_SAMPLES_PER_CLASS = 1000 #number of images for training
NUM_TEST_SAMPLES_PER_CLASS = 200 # number of images for testin
NUM_VALIDATION_SAMPLES_PER_CLASS = 200 #number of images for validation
import torch
import numpy as np
import random

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.mps.manual_seed(SEED)

def set_up_vit():
    print('loading clip_vit')
    clip_vit = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14", 
                device_map="auto", 
                low_cpu_mem_usage=True)
    image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    if torch.cuda.is_available():
        clip_vit.cuda()
    else:
        clip_vit.to('cpu')
    print('finish loading clip_vit of type ', type(clip_vit))
    return [clip_vit, image_processor]

def save_checkpoint(model: FusionModel, inference: Inference, path:str):
    branch_dims = {}
    freq_branch = None
    for branch in model.branches:
        if not (isinstance(branch, Branch)):
            raise TypeError('Expected instance of Branch but got ', type(branch))
        branch_dims[branch.get_name()] = branch.get_dim()
        freq_branch = branch if branch.get_name() == 'frequency' else freq_branch

    if not freq_branch:
        raise RuntimeError('Frequency branch is not found')
    
    torch.save({
        'freq_state': freq_branch.state_dict(),
        'head_state': model.classifier_head.state_dict(),
        'branch_dims': branch_dims,
        'calibrator': inference.calibrator
    }, path)

@torch.no_grad()
def precompute_features(model: FusionModel, preprocessing: Preprocessing, dataset, device, out_path):
    """Compute all features and save it to a .pt file for access in later parts, 
    instead of having to extract features one by one across different epochs -> train data
    remains the same -> features extracted are identical
    Since samples are extracted from index 0 to index NUM_SAMPLES -> if number of samples differ -> stale
    Re-extracts if number of samples do not match"""
    out_path = Path(out_path)
    current_n = len(dataset)

    if out_path.exists():
        cached = torch.load(out_path)
        cached_n = len(cached['labels'])
        if cached_n == current_n:
            print(f"Cache hit ({out_path}): {cached_n} samples match. Skipping extraction.")
            return
        else:
            print(f"Cache stale ({out_path}): cached={cached_n}, current={current_n}. Re-extracting.")

    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        collate_fn=preprocessing.collate_fn,
                        num_workers=0, shuffle=False)

    all_feats = {branch.get_name(): [] for branch in model.branches}
    all_labels = []

    for pixel_dict, labels in loader:
        pixel_dict, labels = to_mps(pixel_dict, labels)
        for branch in model.branches:
            px = branch(pixel_dict[branch.get_name()])
            all_feats[branch.get_name()].append(px.cpu())
        all_labels.append(labels.cpu())

    cached = {name: torch.cat(feats, dim=0) for name, feats in all_feats.items()}
    cached['labels'] = torch.cat(all_labels, dim=0)
    torch.save(cached, out_path)

def train_model():
    start_time = time.perf_counter()
    #CLIP ViT set-up
    clip_vit, vit_image_processor = set_up_vit()
    device = 'mps' if torch.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    #instantiate necessary variables
    branches = [CLIPFeatureBranch(clip_model=clip_vit, image_processor=vit_image_processor),
                FrequencyBranch(output_dim=256)]
    preprocessing = Preprocessing(branches=branches)
    data_fetcher = DataFetch(preprocessing=preprocessing)
    model = FusionModel(branches=branches, num_classes = 1)
    if torch.mps.is_available():
        model.to(device)

    # # --- one-time feature caching ---
    train_raw = data_fetcher.fetch_data(num_samples=NUM_TRAIN_SAMPLES_PER_CLASS, train=True)
    val_raw   = data_fetcher.fetch_data(num_samples=NUM_VALIDATION_SAMPLES_PER_CLASS, val=True)
    test_raw  = data_fetcher.fetch_data(num_samples=NUM_TEST_SAMPLES_PER_CLASS, test=True, 
                                        start_n = NUM_VALIDATION_SAMPLES_PER_CLASS)
    test_robust = data_fetcher.fetch_data(num_samples=NUM_TEST_SAMPLES_PER_CLASS, test=True, test_robust=True, 
                                        start_n = NUM_VALIDATION_SAMPLES_PER_CLASS)

    precompute_features(model, preprocessing, train_raw, device, TRAIN_FEATS_PATH)
    print('pc1')
    precompute_features(model, preprocessing, val_raw,   device, VAL_FEATS_PATH)
    print('pc2')
    precompute_features(model, preprocessing, test_raw,  device, TEST_FEATS_PATH)
    print('pc3')
    precompute_features(model, preprocessing, test_robust, device, TEST_ROBUST_FEATS_PATH)
    # --- fast cached loaders ---
    train_loader = DataLoader(CachedFeatureDataset(TRAIN_FEATS_PATH), batch_size=BATCH_SIZE,
                               collate_fn=cached_collate_fn, shuffle=True)
    
    val_loader   = DataLoader(CachedFeatureDataset(VAL_FEATS_PATH), batch_size=BATCH_SIZE,
                               collate_fn=cached_collate_fn, shuffle=False)
    test_loader  = DataLoader(CachedFeatureDataset(TEST_FEATS_PATH), batch_size=BATCH_SIZE,
                               collate_fn=cached_collate_fn, shuffle=False)
    test_robust_loader = DataLoader(CachedFeatureDataset(TEST_ROBUST_FEATS_PATH), batch_size=BATCH_SIZE,
                               collate_fn=cached_collate_fn, shuffle=False)
    # #create training reqs 
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,  # smaller LR than training from scratch, since we're fine-tuning pretrained weights
        weight_decay=1e-2,
    )
    print("RIGHT AFTER optimizer")
    criterion = nn.BCEWithLogitsLoss()
    print("RIGHT AFTER criterion")

    #create loss lists
    loss_per_epoch_val = []
    loss_per_epoch_train = []

    #feature extraction & training loop
    for epoch in range(5): 
        model.train()
        total_loss = 0 
        print("BEFORE INNER LOOP AT", epoch)
        for i, [pixel_dict, labels] in enumerate(train_loader):
            #refactor 
            labels = labels.unsqueeze(1).float()
            pixel_dict, labels = to_mps(pixel_dict, labels)
            outputs = model.forward_cached(pixel_dict) #predict
            loss = criterion(outputs, labels) #error / likelihood of wrong
            loss.backward() #accumulate gradients
            optimizer.step() #adjust model weights

            total_loss += loss.item()
            optimizer.zero_grad() #reset grad acc
            print("END for  batch with total loss ", total_loss)

        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch+1}, train loss: {avg_loss:.4f}")
        loss_per_epoch_train.append(avg_loss)
        avg_loss_val = Inference.check_entropy_loss(model, val_loader=val_loader)
        loss_per_epoch_val.append(avg_loss_val)
    
    print("Validation loss per epoch:", loss_per_epoch_val, 
          "Training loss per epoch:", loss_per_epoch_train)
    val_inf = Inference(model=model, device=device)
    val_inf.fit_calibrator(val_loader=val_loader)
    
    test_inference = Inference(model=model, device=device)
    roc_auc_clean, cm_clean= test_inference.inference_results(test_loader = test_loader)
    roc_auc_robust, cm_robust = test_inference.inference_results(test_loader = test_robust_loader)
    print("ROC_AUC_clean:" ,roc_auc_clean)
    print("Confusion matrix clean", cm_clean)
    print("ROC_AUC_robust:" ,roc_auc_robust)
    print("Confusion matrix robust", cm_robust)

    save_checkpoint(model, path = CHECKPOINT_PATH, inference=val_inf) #save model to a checkpoint path
    end_time = time.perf_counter()
    print(f"Runtime: {end_time - start_time:.3f} seconds")

if __name__ == '__main__':
    train_model()