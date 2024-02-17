import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


def load_data() -> DataLoader:
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load dataset
    dataset = torchvision.datasets.ImageFolder(root='path_to_your_data', transform=transform)

    # Define DataLoader
    dataloader = DataLoader(dataset, batch_size=15, shuffle=True)

    return dataloader

def train_model(dataloader: DataLoader, epochs: int = 100) -> nn.Module:
    # Load pretrained ResNet-18 model
    model = resnet18(weights = ResNet18_Weights.DEFAULT)

    # Modify the last fully connected layer for 8 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            optimiser.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimiser.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

