import torch
import torch.nn as nn
from typing import Tuple, List


class CNN(nn.Module):
    """
    Convolutional Neural Network that takes in an image and produces a feature vector.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int] = (30, 30),
        num_conv_blocks: int = 2,
        filters_list: List[int] = [16, 32],
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        feature_vector_output_size: int = 1,
    ):
        """
        Args:
            output_size (int): The number of outputs
            image_shape (Tuple[int, int]): Shape of the input image
        """
        super(CNN, self).__init__()
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
            image_shape[0] // (self.maxpool_stride**self.num_conv_blocks),
            image_shape[1] // (self.maxpool_stride**self.num_conv_blocks),
        )

        self.feature_vector = nn.Linear(
            filters_list[-1]
            * output_size_after_max_pools[0]
            * output_size_after_max_pools[1],
            feature_vector_output_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Imaege of shape (n, 1, image_shape[0], image_shape[1])

        Returns: torch.Tensor of size (n, output_size)
        """
        for i in range(self.num_conv_blocks):
            x = self.conv_blocks[i](x)
            x = self.relu(x)
            x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.feature_vector(x)
        return x


class DynamicCFN(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int] = (30, 30),
        num_conv_blocks: int = 2,
        conv_filters_list: List[int] = [16, 32],
        conv_kernel_size: int = 3,
        conv_stride: int = 1,
        conv_padding: int = 1,
        feature_vector_output_size: int = 10,
        images_per_sequence: int = 4,
        metadata_size: int = 8,
        hidden_mlp_layers: int = 2,
        hidden_mlp_size: int = 64,
    ):
        super(DynamicCFN, self).__init__()
        self.image_shape = image_shape
        self.images_per_sequence = images_per_sequence
        self.feature_vector_size = feature_vector_output_size
        self.num_conv_blocks = num_conv_blocks
        self.conv_filters_list = conv_filters_list
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding

        self.metadata_size = metadata_size
        self.use_metadata = metadata_size > 0

        self.hidden_mlp_layers = hidden_mlp_layers
        self.hidden_mlp_size = hidden_mlp_size

        self.cnn = CNN(
            image_shape=image_shape,
            num_conv_blocks=num_conv_blocks,
            filters_list=conv_filters_list,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            feature_vector_output_size=feature_vector_output_size,
        )

        self.mlp = MLP(
            images_per_sequence * feature_vector_output_size + metadata_size,
            1,
            hidden_size=hidden_mlp_size,
            hidden_layers=hidden_mlp_layers,
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x (Tuple[torch.Tensor, torch.Tensor]): Tuple containing images and metadata (if used)

        images: Concatenated images of shape (n, 1, images_per_sequence * image_shape[0], image_shape[1])
        metadata: Metadata of shape (n, metadata_size)

        Returns: Prediction for the asteroid candidate(s) (n, 1)
        """
        if self.use_metadata:
            images, metadata = x
        else:
            images = x
        # Images is the concatenation of the images, split back into individual images
        images = torch.split(images, self.image_shape[0], dim=2)
        feature_vectors = [self.cnn(image) for image in images]
        if self.use_metadata:
            feature_vectors.append(metadata)
        feature_vector = torch.cat(feature_vectors, dim=1)

        # Fusion
        x = self.mlp(feature_vector)
        return x


# Convolutional Fusion Network
class CFN(nn.Module):
    """
    Neural Network that takes in a set of images and predicts whether they represent an asteroid.
    """

    def __init__(
        self,
        images_per_sequence: int,
        feature_vector_size: int,
        image_shape: Tuple[int, int],
    ):
        """
        Args:
            images_per_sequence (int): Number of images per sequence
            feature_vector_size (int): Size of the hidden representation
            image_shape (Tuple[int, int]): Width and height of the individual images
        """
        super(CFN, self).__init__()
        self.image_shape = image_shape
        self.images_per_sequence = images_per_sequence
        self.feature_vector_size = feature_vector_size
        # CNN for the images, ouputs a feature vector
        self.cnn = CNN(feature_vector_output_size=feature_vector_size)
        # Merge the feature vectors to a single label
        self.merge = nn.Linear(images_per_sequence * feature_vector_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Concatenated images of shape (n, 1, images_per_sequence * image_shape[0], image_shape[1])

            Returns: Prediction for the asteroid candidate(s) (n, 1)
        """
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
        self.layers = nn.ModuleList()
        output_layer_input_size = input_size
        if hidden_layers > 0:
            self.layers.extend([nn.Linear(input_size, hidden_size)])
            self.layers.extend(
                [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
            )
            output_layer_input_size = hidden_size

        self.output_layer = nn.Linear(output_layer_input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x


# Meta Convolutional Fusion Network
class MCFN(nn.Module):
    def __init__(
        self,
        images_per_sequence: int = 4,
        feature_vector_size: int = 10,
        image_shape: Tuple[int, int] = (30, 30),
        metadata_size: int = 8,  # Assuming just using positions as default
        hidden_mlp_layers: int = 2,
        hidden_mlp_size: int = 64,
    ):
        super(MCFN, self).__init__()
        self.image_shape = image_shape
        self.images_per_sequence = images_per_sequence
        self.feature_vector_size = feature_vector_size
        # CNN for the images, ouputs a feature vector
        self.cnn = CNN(feature_vector_output_size=feature_vector_size)
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

        Returns: Prediction for the asteroid candidate(s) (n, 1)
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
    test_model_no_metadata = DynamicCFN(metadata_size=0, hidden_mlp_layers=0)
    print(test_model_no_metadata)

    old_model = CFN(4, 10, (30, 30))
    print(old_model)

    # Comparing new Dynamic Model to old model
    test_model_with_metadata = DynamicCFN(metadata_size=8)
    print(test_model_with_metadata)

    old_model = MCFN()
    print(old_model)
