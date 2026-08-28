#python code for backend / model training in this folder
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from eval import create_test_loader, evaluate_model
from fetch_data import fetch_test_data, fetch_train_data, load_subset_dataset
from preprocessing import collate_fn

num_workers = 2
batch_size = 64

def main(): 
    dataset = load_subset_dataset(fetch_train_data(), 1000)
    cnn_model = resnet50(weights=ResNet50_Weights.DEFAULT) #Model used is here
    if torch.cuda.is_available(): 
        cnn_model = cnn_model.cuda()
    for param in cnn_model.parameters(): #freeze first for baselin -> if not might overfit
        param.requires_grad = False
    
    num_classes = 2
    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, num_classes)

    print("RIGHT BEFORE TRAIN LOADER")

    train_loader = DataLoader(dataset=dataset, 
                              batch_size=batch_size, 
                              collate_fn=collate_fn, 
                              num_workers=num_workers, 
                              shuffle=True)

    print("RIGHT AFTER TRAIN LOADER")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, cnn_model.parameters()),
        lr=1e-4  # smaller LR than training from scratch, since we're fine-tuning pretrained weights
    )
    print("RIGHT AFTER optimizer")

    criterion = nn.CrossEntropyLoss()
    print("RIGHT AFTER criterion")

    validation_set = load_subset_dataset(dataset=fetch_test_data(), num_samples=200)
    test_loader = create_test_loader(validation_set)
    print("RIGHT AFTER test loader")

    #TODO: Implement a EarlyStopper instead of fixed epoch, with a max EPOCH value 
    # => Feature extraction & training loop 
    for epoch in range(5): 
        cnn_model.train()
        total_loss = 0 
        print("BEFORE INNER LOOP AT", epoch)
        for i, [pixel_values, labels] in enumerate(train_loader):
            print("batch", i)
            if torch.cuda.is_available(): 
                pixel_values, labels = pixel_values.cuda(), labels.cuda()

            outputs = cnn_model(pixel_values) #predict
            loss = criterion(outputs, labels) #error / likelihood of wrong
            loss.backward() #accumulate gradients
            optimizer.step() #adjust model weights

            total_loss += loss.item()
            optimizer.zero_grad() #reset grad acc
            print("END for  batch with total loss ", total_loss)

        print(f"Epoch {epoch+1}, train loss: {total_loss/len(train_loader):.4f}")
        evaluate_model(cnn_model, test_loader=test_loader)

if __name__ == '__main__':
    main()