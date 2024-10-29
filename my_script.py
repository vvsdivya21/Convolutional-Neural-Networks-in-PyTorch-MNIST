import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

# The values 0.1307 and 0.3081 used for the Normalize() transformation below are the global mean and standard 
# deviation of the MNIST dataset. This is equivalent to scaling all pixel values between [0, 1].
transform = torchvision.transforms.Compose([
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
            ])

train_data = torchvision.datasets.MNIST('', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST('', train=False, download=True, transform=transform)

# We can see some information about this data, including the transform we've applied.
train_data

# Visualize the first 16 elements of the training set
fig = plt.figure(figsize=(10, 10))
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.imshow(train_data.data[i], cmap='gray')


# Define the model
class CNN(torch.nn.Module):
    # Your code here
    raise NotImplementedError() # remove this line


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        ###########################################################################################################
        # Layer 1-  28 filters of size 5x5x1
        self.conv1 = nn.Conv2d(1, 28, kernel_size=5, stride=1, padding=2)  
        nn.init.normal_(self.conv1.weight, mean=0, std=0.05)  
        nn.init.constant_(self.conv1.bias, 0.1) 
        # Activation_function
        self.relu1 = nn.ReLU()
        # Max pooling layer : pool size 2x2 and stride 2
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        ###########################################################################################################
        # Layer 2- 16 filters of size 5x5x28
        self.conv2 = nn.Conv2d(28, 16, kernel_size=5, stride=1, padding=0)  # No padding
        nn.init.normal_(self.conv2.weight, mean=0, std=0.05)
        nn.init.constant_(self.conv2.bias, 0.1)
        # Activation_function
        self.relu2 = nn.ReLU()
        # Max pooling layer : pool size 2x2 and stride 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten layer
        self.flatten = nn.Flatten()
        # Fully connected layer
        self.fc1 = nn.Linear(16 * 5 * 5, 1024)
        # Activation_function
        self.relu3 = nn.ReLU()

        ###########################################################################################################
        # Another fully connected layer with 128 units
        self.fc2 = nn.Linear(1024, 128)
        # Activation_function
        self.relu4 = nn.ReLU()

        ###########################################################################################################
        # Dropout layer for overfitting
        self.dropout = nn.Dropout(p=0.2)
        
        
        ###########################################################################################################
        ###########################################################################################################
        # Final fully connected layer and log softmax activation
        self.fc3 = nn.Linear(128, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        
    #Function2
    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.dropout(x)
        x = self.log_softmax(self.fc3(x))
        return x


model = CNN()
summary(model, (1, 28, 28))  # Input shape: (channels, height, width)

###########################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

###########################################################################################################
# Read the dataset
###########################################################################################################

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

###########################################################################################################
#train-test split 
###########################################################################################################
train_indices, val_indices = train_test_split(list(range(len(mnist_dataset))), test_size=0.1, random_state=42)
train_dataset = torch.utils.data.Subset(mnist_dataset, train_indices)
val_dataset = torch.utils.data.Subset(mnist_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

###########################################################################################################
#Defining the loss function
###########################################################################################################
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

#setting the num_epochs as 10
num_epochs = 10  
for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    correct_train = 0
    total_train = 0
###########################################################################################################
    #loop the train_loader
###########################################################################################################
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets).sum().item()
#################################################################################################################
#after 100 batches, print the statistics
#################################################################################################################

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/(batch_idx+1):.3f}')

#############################################################################################################
    #ACCURACY
    train_accuracy = correct_train / total_train
#############################################################################################################

    model.eval()  
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            _, val_predicted = val_outputs.max(1)
            total_val += val_targets.size(0)
            correct_val += val_predicted.eq(val_targets).sum().item()

    val_accuracy = correct_val / total_val

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/(batch_idx+1):.3f}, '
          f'Train Accuracy: {100 * train_accuracy:.2f}%, Val Accuracy: {100 * val_accuracy:.2f}%')
############################################################################################################
#TESTING
############################################################################################################
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()  
correct_test = 0
total_test = 0

with torch.no_grad():
    for test_inputs, test_targets in test_loader:
        test_outputs = model(test_inputs)
        _, test_predicted = test_outputs.max(1)
        total_test += test_targets.size(0)
        correct_test += test_predicted.eq(test_targets).sum().item()
#########################################################################################################
#ACCURACY
#########################################################################################################
test_accuracy = correct_test / total_test
print(f'Test Accuracy: {100 * test_accuracy:.2f}%')

# Your code here
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


torch.manual_seed(42)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

########################################################################################################
#TRAIN-TEST SPLIT
########################################################################################################
train_indices, val_indices = train_test_split(list(range(len(mnist_dataset))), test_size=0.1, random_state=42)
train_dataset = torch.utils.data.Subset(mnist_dataset, train_indices)
val_dataset = torch.utils.data.Subset(mnist_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)


num_epochs = 50

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets).sum().item()
########################################################################################################
#CALCULATE ACCURACY AND LOSS
########################################################################################################

    train_accuracy = correct_train / total_train
    train_losses.append(running_loss / (batch_idx + 1))
    train_accuracies.append(train_accuracy)
########################################################################################################
#VALIDATION
########################################################################################################
    model.eval()  
    correct_val = 0
    total_val = 0
    val_loss = 0.0

    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            _, val_predicted = val_outputs.max(1)
            total_val += val_targets.size(0)
            correct_val += val_predicted.eq(val_targets).sum().item()
            val_loss += criterion(val_outputs, val_targets).item()
########################################################################################################
#CALCULATE LOSS
########################################################################################################
    val_accuracy = correct_val / total_val
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {train_losses[-1]:.3f}, Train Accuracy: {100 * train_accuracy:.2f}%, '
          f'Val Loss: {val_losses[-1]:.3f}, Val Accuracy: {100 * val_accuracy:.2f}%')

########################################################################################################
#PLOTTING TRAINING AND VALIDATION LOSS
########################################################################################################
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
########################################################################################################
#PLOTTING TRAINING AND VALIDATION ACCURACY
########################################################################################################
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#########################################################################################################
#TESTING
#########################################################################################################
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval() 
correct_test = 0
total_test = 0

with torch.no_grad():
    for test_inputs, test_targets in test_loader:
        test_outputs = model(test_inputs)
        _, test_predicted = test_outputs.max(1)
        total_test += test_targets.size(0)
        correct_test += test_predicted.eq(test_targets).sum().item()

###########################################################################################################
#ACCURACY
###########################################################################################################
test_accuracy = correct_test / total_test
print(f'Test Accuracy: {100 * test_accuracy:.2f}%')


print("start")
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


optimizer = optim.Adam(model.parameters(), lr=1e-2)


criterion = nn.CrossEntropyLoss()

num_epochs = 3  
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_correct_train = 0
    total_samples_train = 0
    epoch_loss_train = 0.0
    
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        total_correct_train += (predicted == labels).sum().item()
        total_samples_train += labels.size(0)
        epoch_loss_train += loss.item()
    
    model.eval()
    total_correct_val = 0
    total_samples_val = 0
    epoch_loss_val = 0.0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            total_correct_val += (predicted == labels).sum().item()
            total_samples_val += labels.size(0)
            epoch_loss_val += loss.item()
    
    avg_loss_train = epoch_loss_train / len(train_loader)
    avg_loss_val = epoch_loss_val / len(val_loader)
    train_accuracy = total_correct_train / total_samples_train
    val_accuracy = total_correct_val / total_samples_val
    
    train_losses.append(avg_loss_train)
    val_losses.append(avg_loss_val)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
total_correct_test = 0
total_samples_test = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_correct_test += (predicted == labels).sum().item()
        total_samples_test += labels.size(0)

test_accuracy = total_correct_test / total_samples_test
print("Accuracy on the Test Set:", test_accuracy)
