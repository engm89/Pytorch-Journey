import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.datasets as datasets
import torchaudio.transforms as transforms
import torchvision.datasets
from torch.utils.data import DataLoader

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_class=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_class)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x= x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_channel = 1
num_classes =10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

train_dataset = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./dataset', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])), batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./dataset', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])), batch_size=batch_size, shuffle=True)

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_dataset):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    
    num_correct =0
    num_samples =0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions ==y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuacy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


    