import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
from utils.utils import to_device
from fusion_model import FusionModel

CLASSES = {0: 'REAL', 1:'FAKE'}
REAL_IDX = 0
FAKE_IDX = 1

@torch.no_grad()
def check_entropy_loss(model: nn.Module, val_loader: DataLoader):
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    total_loss = 0.0

    criterion = nn.BCEWithLogitsLoss()

    for i, [pixel_dict, labels] in enumerate(val_loader):
        print("batch", i)
        labels = labels.unsqueeze(1).float() 
        if torch.cuda.is_available(): 
            pixel_dict = {k: v.cuda() for k, v in pixel_dict.items()}
            labels = labels.cuda()
        outputs = model(pixel_dict)
        loss = criterion(outputs, labels)

        #loss function
        total_loss += loss.item()
    
    avg_loss = total_loss/len(val_loader)
    print("Validation loss" , avg_loss)
    return avg_loss

#this method should be applied on a separate chunk of dataset (test dataset)
@torch.no_grad()
def run_inference(model: FusionModel, test_loader: DataLoader):
    model.eval()
    results = []

    for pixel_dict, labels in test_loader:
        labels = labels.unsqueeze(1).float() 
        pixel_dict, labels = to_device(pixel_dict=pixel_dict, labels=labels)

        logits = model(pixel_dict)
        probs_fake = torch.sigmoid(logits)
        predicted = (probs_fake >= 0.5).long().squeeze(1)
        
        for i in range(labels.size(0)):
            print(labels[i].item(), predicted[i].item())
            results.append({
                'true_label': CLASSES[int(labels[i].item())],
                'predicted_label': CLASSES[int(predicted[i].item())],
                'prob_fake': probs_fake[i].item(),
            })

    all_pred_probs = [res['prob_fake'] for res in results]
    all_pred_labels = [res['predicted_label'] for res in results]
    all_true_labels = [res['true_label'] for res in results]

    roc_auc = roc_auc_score(y_true=all_true_labels, y_score=all_pred_probs)
    cm = confusion_matrix(y_true=all_true_labels, y_pred=all_pred_labels)

    return roc_auc, cm