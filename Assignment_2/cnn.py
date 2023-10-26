import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as function
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

class CNN:
    @staticmethod
    def create():
        model = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(6, 16, 5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Flatten(),
            nn.Linear(400, 128),  # Set to 128 units
            nn.ReLU(),
            nn.Linear(128, 128),  # Set to 128 units
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        return model

    @staticmethod
    def validate(model, data):
        correct = 0
        for i, (images, labels) in enumerate(data):
            images = images.to("cpu")  # Move images to CPU
            x = model(images)
            value, pred = torch.max(x, 1)
            correct += (pred == labels).sum().item()
        accuracy = 100 * correct / len(data.dataset)
        return accuracy

    @staticmethod
    def train(trainloader, testloader, num_epoch=3, lr=1e-3, device="cpu"):
        accuracies = []
        cnn = CNN.create().to(device)
        cec = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn.parameters(), lr=lr)
        max_accuracy = 0
        best_model = None

        for epoch in range(num_epoch):
            for i, (images, labels) in enumerate(trainloader):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                pred = cnn(images)
                loss = cec(pred, labels)
                loss.backward()
                optimizer.step()
            accuracy = float(CNN.validate(cnn, testloader))
            accuracies.append(accuracy)
            if accuracy > max_accuracy:
                best_model = copy.deepcopy(cnn)
                max_accuracy = accuracy
            print("Epoch", epoch + 1, ": Accuracy ", accuracy, "%")
        plt.plot(accuracies)

        return best_model
    
    
fashion_cnn = CNN.train(trainloader_fashion, testloader_fashion, 20, device = device)

cifar_cnn = CNN.train(trainloader_cifar, testloader_cifar, 20, device = device)
