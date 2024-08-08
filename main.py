from torchvision.datasets import MNIST
from torch import optim
import torch.nn as nn
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
#local
from pre_process import make_pairs, SiamenseDataset, data_loader
from net import SiameseNet
import numpy as np



train_data = MNIST(
    './mnistdataset',
    train = True,
    download = True,
)

test_data = MNIST(
    './mnistdataset',
    train = False,
    download = True,
)


pairs_train, labels_train = make_pairs(train_data.data,  train_data.targets)
pairs_test, labels_test = make_pairs(test_data.data,  test_data.targets)

train_dataset = SiamenseDataset(pairs_train, labels_train, transform=ToTensor())
test_dataset = SiamenseDataset(pairs_test, labels_test, transform=ToTensor())


train_loader = data_loader(train_dataset, batch_size=1)
test_loader = data_loader(test_dataset)
    


num_epochs = 2
model = SiameseNet()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) 

train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    # Training Phase
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (img1, img2, label) in enumerate(train_loader):
        img1, img2, label = img1.float(), img2.float(), label.float()
        
        optimizer.zero_grad()
        output = model(img1, img2)
        loss = criterion(output, label.unsqueeze(1))
        print('output:',output, 'label:', label.unsqueeze(1), 'loss:',loss)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * img1.size(0)
        # Compute accuracy
        predicted = (output > 0.5).float()  # Binarize output
        correct += (predicted.squeeze() == label).sum().item()
        total += label.size(0)
        
        if i==5:
            break
        
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")


torch.save(model.state_dict(), 'siamese_model.pth')


    