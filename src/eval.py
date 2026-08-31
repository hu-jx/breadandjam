import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
from utils.utils import to_mps
from branches.fusion_model import FusionModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve



CLASSES = {0: 'REAL', 1:'FAKE'}
REAL_IDX = 0
FAKE_IDX = 1

class Inference:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.calibrator = None
    
    @torch.no_grad()
    @staticmethod
    def check_entropy_loss(model: FusionModel, val_loader: DataLoader):
        if torch.cuda.is_available():
            model = model.cuda()
        elif torch.backends.mps.is_available():
            model.to('mps')
        model.eval()

        total_loss = 0.0

        criterion = nn.BCEWithLogitsLoss()

        for i, [pixel_dict, labels] in enumerate(val_loader):
            labels = labels.unsqueeze(1).float() 
            outputs = model.forward_cached(pixel_dict)
            # outputs = model(pixel_dict)
            loss = criterion(outputs, labels)

            #loss function
            total_loss += loss.item()
        
        avg_loss = total_loss/len(val_loader)
        print("Validation loss" , avg_loss)
        return avg_loss

    #this method should be applied on a separate chunk of dataset (test dataset)
    @torch.no_grad()
    def _get_probs(self,data_loader: DataLoader):
        self.model.eval()
        results = []

        for pixel_dict, labels in data_loader:
            labels = labels.unsqueeze(1).float() 
            pixel_dict, labels = to_mps(pixel_dict=pixel_dict, labels=labels)

            logits = self.model.forward_cached(pixel_dict)
            # logits = model(pixel_dict)
            probs_fake = torch.sigmoid(logits)
            predicted = (probs_fake >= 0.9949228167533875).long().squeeze(1)
            
            for i in range(labels.size(0)):
                results.append({
                    'true_label': int(labels[i].item()),
                    'predicted_label': int(predicted[i].item()),
                    'prob_fake': probs_fake[i].item(),
                })

        all_pred_labels = [res['predicted_label'] for res in results]
        all_true_labels = [res['true_label'] for res in results]
        probs = [res['prob_fake'] for res in results]
        return probs, all_pred_labels, all_true_labels
        

    def get_best_threshold(self, data_loader):
        probs, pred_lab, true_lab = self._get_probs(data_loader=data_loader)
        fpr, tpr, thresholds = roc_curve(true_lab, probs)
        youden_j = tpr - fpr
        best_threshold = thresholds[youden_j.argmax()]
        print("Best threshold:", best_threshold)

    def fit_calibrator(self, val_loader):
        """Fit calibrator with the validation set"""
        val_probs, val_true_labels, val_pred_labels= self._get_probs(val_loader)
        val_probs = np.reshape(val_probs, (-1, 1)).tolist()
        self.calibrator = LogisticRegression()
        self.calibrator.fit(val_probs, val_true_labels)
        return val_probs, val_true_labels, val_pred_labels
    
    def calibrate(self, raw_prob):
        if not self.calibrator:
            raise RuntimeError('Calibration has not been completed')
        return self.calibrator.predict(raw_prob)

    def inference_results(self, test_loader):
         probs, pred_labels, true_labels = self._get_probs(test_loader)
         roc_auc = roc_auc_score(y_true=true_labels, y_score=probs)
         cm = confusion_matrix(y_true=true_labels, y_pred=pred_labels)
         tn, fp, fn, tp = cm.ravel()
        
         print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")
         return roc_auc, cm

