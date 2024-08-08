from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class SiamenseDataset(Dataset):
    def __init__(self, pairs, labels, transform= None) -> None:
        self.pairs = pairs
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, index):
        img1 = self.pairs[index, 0]
        img2 = self.pairs[index, 1]
        label = self.labels[index]
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, label 
        
   
    def __len__(self):
        return len(self.pairs)
           

def make_pairs(x, y):

    pairs_lst = []
    labels_lst = []
    num_classes = torch.max(y) + 1

    digit_indices = [np.where(i == y)[0] for i in range(int(num_classes))]

    for idx1 in range(len(x)):
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = np.random.choice(digit_indices[label1])
        x2 = x[idx2]
        
        pairs_lst += [[x1, x2]]
        labels_lst +=[1]
        
        
        label2 = np.random.randint(0, num_classes - 1)
        while label1 == label2:
            label2 = np.random.randint(0, num_classes - 1)

        idx2 = np.random.choice(digit_indices[label2])
        x2 = x[idx2]
    
        pairs_lst += [[x1, x2]]
        labels_lst += [0]       
         
    return np.array(pairs_lst), np.array(labels_lst)


def plot_checker(pair, label , variable) -> None:
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(label[variable])
    axes[0].imshow(pair[variable, 0])
    axes[1].imshow(pair[variable, 1])
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def data_loader(dataset, batch_size=64, shuffle= True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)