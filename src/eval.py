import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def evaluate_model(model: nn.Module, test_loader: DataLoader):
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, [pixel_dict, labels] in enumerate(test_loader):
            print("batch", i)
            if torch.cuda.is_available(): 
                pixel_dict = {k: v.cuda() for k, v in pixel_dict.items()}
                labels = labels.cuda()
            outputs = model(pixel_dict)
            loss = criterion(outputs, labels)

            #loss function
            total_loss += loss.item()

            #percentage accuracy
            _, predicted = torch.max(outputs, dim=1) # Get index of the highest logit
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print("Validation loss" , total_loss/len(test_loader))
    print("Ratio of correct: total ",correct/total)
