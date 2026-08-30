#python code for backend / model training in this folder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from clip_feature import CLIPFeatureBranch
from branch import Branch
from utils.utils import to_device
from fusion_model import FusionModel
from eval import check_entropy_loss, run_inference
from fetch_data import DataFetch
from preprocessing import Preprocessing
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

NUM_WORKERS = 2
BATCH_SIZE = 64
NUM_TRAIN_SAMPLES = 100 #number of images for training
NUM_TEST_SAMPLES = 20
NUM_VALIDATION_SAMPLES = 20 #number of images for validation
CHECKPOINT_PATH = "checkpoint.pt"

def set_up_vit():
    print('loading clip_vit')
    clip_vit = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14", 
                device_map="auto", 
                low_cpu_mem_usage=True)
    image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_vit.to(device)
    print('finish loading clip_vit of type ', type(clip_vit))
    return [clip_vit, image_processor]

def save_checkpoint(model: FusionModel, path: str):
    branch_dims = {}
    for branch in model.branches:
        if not (isinstance(branch, Branch)):
            raise TypeError('Expected instance of Branch but got ', type(branch))
        branch_dims[branch.get_name()] = branch.get_dim()

    torch.save({
        # extract feature branch parameters with .state_dict()
        'head_state': model.classifier_head.state_dict(),
        'branch_dims': branch_dims,
    }, path)

def train():
    #CLIP ViT set-up
    clip_vit, vit_image_processor = set_up_vit()

    #instantiate necessary variables
    branches = [CLIPFeatureBranch(clip_model=clip_vit, image_processor=vit_image_processor)]
    preprocessing = Preprocessing(branches=branches)
    data_fetcher = DataFetch(preprocessing=preprocessing)
    model = FusionModel(branches=branches, num_classes = 1)

    #create data loaders
    train_data = data_fetcher.fetch_data(num_samples=NUM_TRAIN_SAMPLES,train=True)
    print("RIGHT BEFORE TRAIN LOADER")
    train_loader = DataLoader(dataset=train_data, 
                              batch_size=BATCH_SIZE, 
                              collate_fn=preprocessing.collate_fn, 
                              num_workers=NUM_WORKERS, 
                              shuffle=True)
    print("RIGHT AFTER TRAIN LOADER")

    val_data = data_fetcher.fetch_data(num_samples=NUM_VALIDATION_SAMPLES, val=True)
    val_loader = DataLoader(dataset=val_data, 
                              batch_size=BATCH_SIZE, 
                              collate_fn=preprocessing.collate_fn, 
                              num_workers=NUM_WORKERS, 
                              shuffle=False)
    print("RIGHT AFTER val loader")

    print("TEST LOADER")
    test_data = data_fetcher.fetch_data(num_samples=NUM_TEST_SAMPLES, test=True)
    test_loader = DataLoader(dataset=val_data, 
                              batch_size=BATCH_SIZE, 
                              collate_fn=preprocessing.collate_fn, 
                              num_workers=NUM_WORKERS, 
                              shuffle=False)
    
    #create training reqs 
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4  # smaller LR than training from scratch, since we're fine-tuning pretrained weights
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
            print("batch", i)
            #refactor 
            labels = labels.unsqueeze(1).float() 
            pixel_dict, labels = to_device(pixel_dict, labels)

            outputs = model(pixel_dict) #predict
            loss = criterion(outputs, labels) #error / likelihood of wrong
            loss.backward() #accumulate gradients
            optimizer.step() #adjust model weights

            total_loss += loss.item()
            optimizer.zero_grad() #reset grad acc
            print("END for  batch with total loss ", total_loss)

        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch+1}, train loss: {avg_loss:.4f}")
        loss_per_epoch_train.append(total_loss)
        avg_loss_val = check_entropy_loss(model, val_loader=val_loader)
        loss_per_epoch_val.append(avg_loss_val)
    
    print("Validation loss per epoch:", loss_per_epoch_val, 
          "Training loss per epoch:", loss_per_epoch_train)
    roc_auc, cm = run_inference(model=model, test_loader=test_loader)
    print("ROC_AUC_clean:" ,roc_auc)
    print("Confusion matrix", cm)
    save_checkpoint(model, path = CHECKPOINT_PATH) #save model to a checkpoint path

if __name__ == '__main__':
    train()