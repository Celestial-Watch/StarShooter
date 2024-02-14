import torch
import torch.nn as nn


# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x


# Merge Neural Network
class MNN(nn.Module):
    def __init__(self, images_per_sequence, feature_vector_size, image_shape):
        super(MNN, self).__init__()
        self.image_shape = image_shape
        self.images_per_sequence = images_per_sequence
        self.feature_vector_size = feature_vector_size
        # CNN for the images, ouputs a feature vector
        self.cnn = CNN(feature_vector_size)
        # Merge the feature vectors to a single label
        self.merge = nn.Linear(images_per_sequence * feature_vector_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The input is the concatenation of the images, split back into individual images
        images = torch.split(x, self.image_shape[0], dim=2)
        feature_vectors = [self.cnn(image) for image in images]
        feature_vector = torch.cat(feature_vectors, dim=1)
        x = self.merge(feature_vector)
        x = torch.sigmoid(x)
        return x


def train(
    model: nn.Module,
    x: torch.Tensor,
    y_hat: torch.Tensor,
    criterion: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
):
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y_hat.float())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for every epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    return model

if __name__ == "__main__":
    image_shape = (30, 30)
    images_per_sequence = 4
    feature_vector_size = 10
    sequences = 200
    channels = 1

    # Sequences, channels, images_per_sequence * height, width
    input_sequences = torch.randn(
        sequences, channels, images_per_sequence * image_shape[0], image_shape[1]
    )
    labels = torch.randint(0, 2, (sequences, 1))

    # Create an instance of the MNN model
    model = MNN(images_per_sequence, feature_vector_size, image_shape)

    x = input_sequences
    y_hat = labels

    # Training parameters
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100

    model = train(model, x, y_hat, criterion, optimizer, num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), "model/model.pt")
