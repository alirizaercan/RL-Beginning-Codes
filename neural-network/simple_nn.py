import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 3
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # Second 2D convolutional layer, taking in 32 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        # First fully connected layer
        self.fc1 = nn.Linear(9216, 128)
        # Second fully connected layer 
        self.fc2 = nn.Linear(128, 10)
        
#my_nn = SimpleNN()
#print(my_nn)
        
    # X represents our data
    def forward(self, x):
        # Pass data through conv1 
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        # Run max pooling over x
        x = F.max_pool2d(x, 2)
        # Pass data through dropout1
        x = self.dropout1(x)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through fc1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
        return output
    
random_data = torch.randn(1, 1, 28, 28)
model = SimpleNN()
result = model(random_data)
print(result)
    
