import torch
import torch.nn as nn
from typing import Tuple


# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self, output_size: int):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x


# Convolutional Fusion Network
class CFN(nn.Module):
    def __init__(
        self,
        images_per_sequence: int,
        feature_vector_size: int,
        image_shape: Tuple[int, int],
    ):
        super(CFN, self).__init__()
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


# Meta Convolutional Fusion Network
class MCFN(nn.Module):
    def __init__(
        self,
        images_per_sequence: int,
        feature_vector_size: int,
        image_shape: Tuple[int, int],
        metadata_size: int,
    ):
        super(MCFN, self).__init__()
        self.image_shape = image_shape
        self.images_per_sequence = images_per_sequence
        self.feature_vector_size = feature_vector_size
        # CNN for the images, ouputs a feature vector
        self.cnn = CNN(feature_vector_size)
        # Merge the feature vectors and metadata to a single label
        self.merge = nn.Linear(
            images_per_sequence * feature_vector_size + metadata_size, 1
        )

    def forward(self, x: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Concatenated images of shape (n, 1, images_per_sequence * image_shape[0], image_shape[1])
            metadata (torch.Tensor): Metadata of shape (n, metadata_size)
        """
        # The input is the concatenation of the images, split back into individual images
        images = torch.split(x, self.image_shape[0], dim=2)
        feature_vectors = [self.cnn(image) for image in images]
        feature_vectors.append(metadata)
        feature_vector = torch.cat(feature_vectors, dim=1)

        # Fusion
        x = self.merge(feature_vector)
        x = torch.sigmoid(x)
        return x