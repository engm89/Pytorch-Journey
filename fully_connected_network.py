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
				
				def forward(self, x):
								x = F.relu(self.fc1(x))
								x = self.fc2(x)
								return x


#
# model = NN(784, 10)
# x = torch.randn(64, 784)
# print(model(x).shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
num_class = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

train_dataset = torch.utils.data.DataLoader(
				torchvision.datasets.MNIST('./dataset', train=True, download=True,
				                           transform=torchvision.transforms.Compose([
								                           torchvision.transforms.ToTensor()
				                           ])), batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.DataLoader(
				torchvision.datasets.MNIST('./dataset', train=False, download=True,
				                           transform=torchvision.transforms.Compose([
								                           torchvision.transforms.ToTensor()
				                           ])), batch_size=batch_size, shuffle=True)

model = NN(input_size=input_size, num_classes=num_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
				for batch_idex, (data, targets) in enumerate(train_dataset):
								data = data.to(device=device)
								targets = targets.to(device=device)
								
								data = data.reshape(data.shape[0], -1)
								
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


check_accuracy(train_dataset, model)
check_accuracy(train_dataset, model)
