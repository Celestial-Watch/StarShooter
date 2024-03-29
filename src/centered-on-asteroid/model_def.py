import torch
import torch.nn as nn
import math
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
        feature_vector_output_size: int = 1,
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

        output_shape_after_conv_blocks = image_shape
        for i in range(num_conv_blocks):
            output_shape_after_conv_blocks = self.get_output_after_one_conv(
                output_shape_after_conv_blocks, kernel_size, stride, padding
            )
            output_shape_after_conv_blocks = self.get_output_after_one_maxpool(
                output_shape_after_conv_blocks,
                self.maxpool_kernel_size,
                self.maxpool_stride,
            )

        # We only need one ReLU layer and one maxpool layer, as they don't learn any parameters
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(
            kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride
        )
        # output_size_after_max_pools = (
        #     output_shape_after_conv_blocks[0]
        #     // (self.maxpool_stride**self.num_conv_blocks),
        #     output_shape_after_conv_blocks[1]
        #     // (self.maxpool_stride**self.num_conv_blocks),
        # )

        # print(f"Output size after max pools: {output_size_after_max_pools}")
        # print(
        #     f"feature vector input: {filters_list[-1] * output_size_after_max_pools[0] * output_size_after_max_pools[1]}"
        # )

        print(
            f"Feature vector input size: {filters_list[-1] * output_shape_after_conv_blocks[0] * output_shape_after_conv_blocks[1]}"
        )

        self.feature_vector = nn.Linear(
            filters_list[-1]
            * output_shape_after_conv_blocks[0]
            * output_shape_after_conv_blocks[1],
            feature_vector_output_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"Input shape: {x.shape}")
        for i in range(self.num_conv_blocks):
            x = self.conv_blocks[i](x)
            # print(f"Shape after conv block {i}: {x.shape}")
            x = self.relu(x)
            x = self.maxpool(x)
            # print(f"Shape after max pool {i}: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"Flattened shape: {x.shape}")
        x = self.feature_vector(x)
        return x

    def get_output_after_one_conv(
        self, input_shape: Tuple[int, int], kernel_size: int, stride: int, padding: int
    ) -> Tuple[int, int]:
        """
        Args:
            input_shape (Tuple[int, int]): Shape of the input image
            kernel_size (int): Size of the kernel
            stride (int): Stride of the convolution
            padding (int): Padding of the convolution

        Returns: Shape of the output after one convolution
        """

        return (
            int((input_shape[0] - kernel_size + 2 * padding) / stride + 1),
            int((input_shape[1] - kernel_size + 2 * padding) / stride + 1),
        )

    def get_output_after_one_maxpool(
        self,
        input_shape: Tuple[int, int],
        kernel_size: int,
        stride: int,
        padding: int = 0,
        dilation: int = 1,
    ) -> Tuple[int, int]:
        """
        Args:
            input_shape (Tuple[int, int]): Shape of the input image
            kernel_size (int): Size of the kernel
            stride (int): Stride of the maxpool
            padding (int): Padding of the maxpool
            dilation (int): Dilation of the maxpool

        Returns: Shape of the output after one maxpool
        """
        h_out = math.floor(
            (input_shape[0] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
            + 1
        )
        w_out = math.floor(
            (input_shape[1] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
            + 1
        )

        return (h_out, w_out)


# Convolutional Neural Network
class CNN(nn.Module):
    """
    Convolutional Neural Network that takes in an image and produces a feature vector.
    """

    def __init__(self, output_size: int = 1, image_shape: Tuple[int, int] = (30, 30)):
        """
        Args:
            output_size (int): The number of outputs
            image_shape (Tuple[int, int]): Shape of the input image
        """
        super(CNN, self).__init__()
        n_channels_1 = 16
        n_channels_2 = 32
        final_layer_size = (
            n_channels_2 * (image_shape[0] // 2 // 2) * (image_shape[1] // 2 // 2)
        )
        self.conv1 = nn.Conv2d(1, n_channels_1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            n_channels_1, n_channels_2, kernel_size=3, stride=1, padding=1
        )
        self.fc = nn.Linear(final_layer_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Imaege of shape (n, 1, image_shape[0], image_shape[1])

        Returns: torch.Tensor of size (n, output_size)
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
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

        self.cnn = DynamicCNN(
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
        # print(f"Feature vector shape: {feature_vector.shape}")

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
        self.cnn = DynamicCNN(feature_vector_output_size=feature_vector_size)
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
        self.cnn = DynamicCNN(feature_vector_output_size=feature_vector_size)
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
    test_model_no_metadata = DynamicCFN(
        image_shape=(30, 30),
        num_conv_blocks=1,
        conv_filters_list=[16],
        conv_kernel_size=5,
        feature_vector_output_size=10,
        images_per_sequence=4,
        metadata_size=4,
        hidden_mlp_layers=2,
        hidden_mlp_size=64,
    )
    print(test_model_no_metadata)

    # old_model = CFN(4, 10, (30, 30))
    # print(old_model)

    # # Comparing new Dynamic Model to old model
    # test_model_with_metadata = DynamicCFN(metadata_size=8)
    # print(test_model_with_metadata)

    # old_model = MCFN()
    # print(old_model)
