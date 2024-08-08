from torchvision.datasets import MNIST
from torch import optim
import torch.nn as nn
from torchvision.transforms import ToTensor
import torch
import matplotlib.pyplot as plt
#local
from pre_process import make_pairs, SiamenseDataset, data_loader
from net import SiameseNet
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_seed(42)


test_data = MNIST(
    './mnistdataset',
    train=False,
    download=True,
)

pairs_test, labels_test = make_pairs(test_data.data, test_data.targets)
test_dataset = SiamenseDataset(pairs_test, labels_test, transform=ToTensor())
test_loader = data_loader(test_dataset)

model = SiameseNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the saved model
model.load_state_dict(torch.load('siamese_model.pth'))
model.eval() 


def test(model, test_loader, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2, label = img1.float(), img2.float(), label.float()
            
            output = model(img1, img2)
            loss = criterion(output, label.unsqueeze(1))
            
            running_loss += loss.item() * img1.size(0)
            
            # Compute accuracy
            predicted = (output > 0.5).float()
            correct += (predicted.squeeze() == label).sum().item()
            total += label.size(0)
            
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_accuracy = correct / total

    print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_accuracy:.4f}")



def plot_checker(pair, label, variable, model) -> None:
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        img1 = torch.tensor(pair[variable, 0]).unsqueeze(0).unsqueeze(0).float()
        img2 = torch.tensor(pair[variable, 1]).unsqueeze(0).unsqueeze(0).float()
        output = model(img1, img2)
        prediction = 0 if torch.sigmoid(output).item() > 0.5 else 1
        
        if prediction == label[variable]:
            lb = 'same'
        else:
            lb = 'not same'
            
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(f'Label: {label[variable]} Prediction: {lb}  pred={prediction}, out={torch.sigmoid(output).item()}')
    axes[0].imshow(pair[variable, 0], cmap='gray')
    axes[1].imshow(pair[variable, 1], cmap='gray')
    
    plt.show()
    plt.pause(4)
    plt.close()

    
    
# test(model, test_loader, criterion) #NOTE test model with test dataset

plot_checker(pairs_test, labels_test, variable=5, model=model)  #NOTE  101 label that not same  #85, 87, 89
