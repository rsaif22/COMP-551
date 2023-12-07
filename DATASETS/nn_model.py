import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import copy

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        output = self.softmax(x)
        return output
    
    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x)
            _, predicted = torch.max(output.data, 1)
        return predicted
    
    def fit(self, x, y, learning_rate=0.001, epochs=5, batch_size=32):
        criterion = nn.CrossEntropyLoss()
        opt_params = self.parameters()
        optimizer = torch.optim.Adam(params=opt_params, lr=learning_rate)
        self.train()
        dataset = TensorDataset(x, y)

        best_model = None 
        best_loss = torch.inf
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            # Select random samples
            for batch in dataloader:
                current_x, current_y = batch
                optimizer.zero_grad()
                outputs = self.forward(current_x)
                loss = criterion(outputs, current_y)
                if (loss.item() < best_loss):
                    best_loss = loss.item()
                    best_model = copy.deepcopy(self)
                loss.backward()
                optimizer.step()
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        return best_model



    
