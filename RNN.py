import torch
import torchvision
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transform
from torch.utils.data import DataLoader


input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
train_dataset = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./dataset', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])), batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./dataset', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])), batch_size=batch_size, shuffle=True)


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_dataset):
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuacy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()