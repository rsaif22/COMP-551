import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

class Network(nn.Module):
    def __init__(self, output_classes, input_channels, input_size, inter_channels=[32, 64], kernel_size=5, stride=1):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=inter_channels[0], kernel_size=kernel_size,
        stride=stride)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Formula  = (D + 2*padding - K) / stride + 1
        conv1_output_size = (input_size - kernel_size) // stride + 1
        pool1_output_size = (conv1_output_size) // 2
        self.conv2 = nn.Conv2d(inter_channels[0], inter_channels[1], kernel_size)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv2_output_size = (pool1_output_size - kernel_size) // stride + 1
        pool2_output_size = (conv2_output_size) // 2
        
        self.fc_input_size = inter_channels[1] * pool2_output_size**2
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, output_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1, self.fc_input_size) # Flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class CNN:
    def __init__(self, num_classes, input_channels, input_size, inter_channels=[32, 64], kernel_size=5, stride=1):
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size
        self.cnn = Network(num_classes, input_channels, input_size, inter_channels, kernel_size, stride).to("cuda:0")
        self.loss_func = nn.CrossEntropyLoss()
        self.acc_list = None
        self.test_acc_list = None

    def validate(self, data: DataLoader, device):
        correct = 0
        with torch.no_grad():
            for images, labels in data:
                images = images.to(device)
                labels = labels.to(device)
                x = self.cnn(images)
                value, pred = torch.max(x, 1)
                correct += (pred == labels).sum().item()
        accuracy = float(correct) / float(len(data.dataset))
        return accuracy

    def train(self, trainloader, testloader, num_epochs=5, lr=0.001, device="cuda:0", optimizer="adam", momentum=0.0):
        self.acc_list = np.empty((0,))
        self.test_acc_list = np.empty((0,))
        self.cnn.eval()
        accuracy = self.validate(trainloader, device)
        self.acc_list = np.append(self.acc_list, accuracy)
        accuracy = self.validate(testloader, device)
        self.test_acc_list = np.append(self.test_acc_list, accuracy)
        if optimizer == "adam":
            optimizer = optim.Adam(self.cnn.parameters(), lr=lr)
        elif optimizer == "sgd":
            optimizer = optim.SGD(self.cnn.parameters(), lr=lr, momentum=momentum)
        for epoch in range(num_epochs):
            self.cnn.train()
            for images, labels in trainloader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                pred = self.cnn(images)
                loss = self.loss_func(pred, labels)
                loss.backward()
                optimizer.step()

            self.cnn.eval()
            accuracy = self.validate(trainloader, device)
            self.acc_list = np.append(self.acc_list, accuracy)
            accuracy = self.validate(testloader, device)
            self.test_acc_list = np.append(self.test_acc_list, accuracy)
            print(f"Epoch {epoch+1}, train accuracy: {self.acc_list[-1]}, test accuracy: {self.test_acc_list[-1]}")
    
    def predict(self, data, device="cuda:0"):
        pred = None
        with torch.no_grad():
            for images, labels in data:
                images = images.to(device)
                labels = labels.to(device)
                x = self.cnn(images)
                value, pred = torch.max(x, 1)
        return pred

if __name__=="__main__":
    # generator = torch.Generator().manual_seed(seed)
    cnn = Network(10, 3, 32)
