#python code for backend / model training in this folder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from clip_feature import CLIPFeatureBranch
from fusion_model import FusionModel
from eval import evaluate_model
from fetch_data import DataFetch
from preprocessing import Preprocessing
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

NUM_WORKERS = 2
BATCH_SIZE = 64
NUM_TRAIN_SAMPLES = 6 #number of images for training
NUM_VALIDATION_SAMPLES = 2 #number of images for validation

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


def main():
    #CLIP ViT set-up
    clip_vit, vit_image_processor = set_up_vit()

    #instantiate necessary variables
    branches = [CLIPFeatureBranch(clip_model=clip_vit, image_processor=vit_image_processor)]
    preprocessing = Preprocessing(branches=branches)
    data_fetcher = DataFetch(preprocessing=preprocessing)
    model = FusionModel(branches=branches, num_classes = 2)

    #create data loaders
    train_data = data_fetcher.fetch_data(num_samples=NUM_TRAIN_SAMPLES,train=True)
    print("RIGHT BEFORE TRAIN LOADER")
    train_loader = DataLoader(dataset=train_data, 
                              batch_size=BATCH_SIZE, 
                              collate_fn=preprocessing.collate_fn, 
                              num_workers=NUM_WORKERS, 
                              shuffle=True)
    
    print("RIGHT AFTER TRAIN LOADER")
    validation_set = data_fetcher.fetch_data(train= False, num_samples=NUM_VALIDATION_SAMPLES)
    test_loader = DataLoader(dataset=validation_set, 
                              batch_size=BATCH_SIZE, 
                              collate_fn=preprocessing.collate_fn, 
                              num_workers=NUM_WORKERS, 
                              shuffle=False)
    print("RIGHT AFTER test loader")
    
    #create training reqs 
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4  # smaller LR than training from scratch, since we're fine-tuning pretrained weights
    )
    print("RIGHT AFTER optimizer")
    criterion = nn.CrossEntropyLoss()
    print("RIGHT AFTER criterion")

    #feature extraction & training loop
    #TODO: Implement a EarlyStopper instead of fixed epoch, with a max EPOCH value 
    for epoch in range(5): 
        model.train()
        total_loss = 0 
        print("BEFORE INNER LOOP AT", epoch)
        for i, [pixel_values, labels] in enumerate(train_loader):
            print("batch", i)
            if torch.cuda.is_available(): 
                pixel_values, labels = pixel_values.cuda(), labels.cuda()

            outputs = model(pixel_values) #predict
            loss = criterion(outputs, labels) #error / likelihood of wrong
            loss.backward() #accumulate gradients
            optimizer.step() #adjust model weights

            total_loss += loss.item()
            optimizer.zero_grad() #reset grad acc
            print("END for  batch with total loss ", total_loss)

        print(f"Epoch {epoch+1}, train loss: {total_loss/len(train_loader):.4f}")
        evaluate_model(model, test_loader=test_loader)

if __name__ == '__main__':
    main()