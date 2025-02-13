import torch.nn as nn
import torch

class BinaryClassifier(nn.Module):
    def __init__(self, input_shape):
        super(BinaryClassifier, self).__init__()
        
        # Flatten the input embeddings
        self.flatten = nn.Flatten()
        
        # Define a fully connected layer (Dense) with ReLU activation
        self.fc1 = nn.Linear(input_shape[0] * input_shape[1], 2)  # Adjust the number of units as needed
        self.relu = nn.ReLU()
        
        # Define the output layer for binary classification
        self.fc2 = nn.Linear(2, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x