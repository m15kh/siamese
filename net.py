import torch
import torch.nn as nn


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        
        self.fc2_1 = nn.Linear(1, 128)
        self.fc2_2 = nn.Linear(128, 1)

    def forward_once(self, input):
        x = self.flatten(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        out = self.relu(x)
        return out

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        
        distance = self.euclidean_distance(out1, out2)
        dense_layer = self.fc2_1(distance)
        dense_layer = self.relu(dense_layer)
        output = torch.sigmoid(self.fc2_2(dense_layer))
        
        return output
    
    def euclidean_distance(self, x1 , x2):
        sum_square = torch.sum(torch.square(x1 - x2), dim=1, keepdim=True)  
        return torch.sqrt(torch.maximum(sum_square, torch.tensor(1e-3)))  
    
    
    
