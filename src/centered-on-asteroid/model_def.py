import torch
import torch.nn as nn
from typing import Tuple, List


class DynamicCNN(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int] = (30, 30),
        num_conv_blocks: int = 2,
        filters_list: List[int] = [16, 32],
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        output_size: int = 1,
    ):
        super(DynamicCNN, self).__init__()
        self.image_shape = image_shape

        self.num_conv_blocks = num_conv_blocks
        self.filters_list = filters_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.maxpool_kernel_size = 2
        self.maxpool_stride = 2

        self.conv_blocks = nn.ModuleList(
            [nn.Conv2d(1, filters_list[0], kernel_size, stride, padding)]
        )
        self.conv_blocks.extend(
            [
                nn.Conv2d(
                    filters_list[i], filters_list[i + 1], kernel_size, stride, padding
                )
                for i in range(num_conv_blocks - 1)
            ]
        )

        # We only need one ReLU layer and one maxpool layer, as they don't learn any parameters
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(
            kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride
        )
        output_size_after_max_pools = (
            image_shape[0] // self.maxpool_kernel_size**self.num_conv_blocks,
            image_shape[1] // self.maxpool_kernel_size**self.num_conv_blocks,
        )

        self.feature_vector = nn.Linear(
            filters_list[-1]
            * output_size_after_max_pools[0]
            * output_size_after_max_pools[1],
            output_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_conv_blocks):
            x = self.conv_blocks[i](x)
            x = self.relu(x)
            x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.feature_vector(x)
        return x


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


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        hidden_layers: int = 2,
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.layers.extend(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x


# Meta Convolutional Fusion Network
class MCFN(nn.Module):
    def __init__(
        self,
        images_per_sequence: int,
        feature_vector_size: int,
        image_shape: Tuple[int, int],
        metadata_size: int,
        hidden_mlp_layers: int = 2,
        hidden_mlp_size: int = 64,
    ):
        super(MCFN, self).__init__()
        self.image_shape = image_shape
        self.images_per_sequence = images_per_sequence
        self.feature_vector_size = feature_vector_size
        # CNN for the images, ouputs a feature vector
        self.cnn = DynamicCNN(output_size=feature_vector_size)
        # Merge the feature vectors and metadata to a single label
        self.mlp = MLP(
            images_per_sequence * feature_vector_size + metadata_size,
            1,
            hidden_size=hidden_mlp_size,
            hidden_layers=hidden_mlp_layers,
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x (Tuple[torch.Tensor, torch.Tensor]): Tuple containing images and metadata

        images: Concatenated images of shape (n, 1, images_per_sequence * image_shape[0], image_shape[1])
        metadata: Metadata of shape (n, metadata_size)
        """
        images, metadata = x
        # Images is the concatenation of the images, split back into individual images
        images = torch.split(images, self.image_shape[0], dim=2)
        feature_vectors = [self.cnn(image) for image in images]
        feature_vectors.append(metadata)
        feature_vector = torch.cat(feature_vectors, dim=1)

        # Fusion
        x = self.mlp(feature_vector)
        return x


if __name__ == "__main__":
    # Comparing new Dynamic Model to old model
    test_model = DynamicCNN(num_conv_blocks=3, filters_list=[16, 32, 64])
    print(test_model)

    test_model = CNN(1)
    print(test_model)
