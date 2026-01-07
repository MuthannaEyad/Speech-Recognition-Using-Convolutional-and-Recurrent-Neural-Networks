"""
CNN Module for Speech Recognition

This file defines the convolutional feature extractor used to process 
log-Mel spectrogram's from the audio. The CNN reduces temporal and spectral
dimensions while extracting higher-level features for the downstream RNN.

Design Choices:
- Multiple convolutional layers with ReLU activations are used to learn complex 
  hierarchical features from audio spectrograms.

- Kernel size, stride, and padding are chosen to preserve temporal resolution 
  while gradually reducing spectral dimension.

- MaxPool2d with kernel (2,1) is applied to downsample the time dimension 
  without collapsing the frequency bins completely.

- Increasing channels exponentially (base_channels * 2^i) allows the network
  to capture more feature maps at deeper layers.
"""

from torch import nn


class CNN(nn.Module):
    """
    Convolutional Neural Network for feature extraction from spectrograms.

    What it does:
    - Processes 2D spectrogram images through a series of convolutional blocks.
    - Uses striding in the first layer to downsample the time/frequency dimensions.
    - Employs residual (skip) connections to allow information to flow directly
      across blocks if dimensions match.

    Why:
    - CNNs are excellent at identifying local patterns (like formats or pitch
      changes) regardless of where they occur in the audio clip.
    - GELU activation and BatchNorm2d provide superior stability and faster
      convergence for speech-related spectral data compared to traditional ReLU.
    """

    def __init__(self, num_layers, in_channels, base_channels, kernel_size, stride):
        """
        Initializes the CNN feature extractor architecture.

        What it does:
        - Builds a list of Sequential blocks, each containing Conv2d, BatchNorm,
          GELU, and Dropout.
        - Sets a stride of (2, 2) on the first layer only to compress the input
          resolution early.

        Why:
        - Early striding reduces the sequence length (T) before it hits the RNN,
          dramatically lowering VRAM usage and training time.
        - ModuleList is used to allow for the manual iteration and residual
          logic implemented in the forward pass.

        :param num_layers: Number of convolutional blocks in the network.
        :param in_channels: Number of input channels (usually 1 for spectrograms).
        :param base_channels: Number of channels in the convolutional layers.
        :param kernel_size: Kernel size for the filters (height, width).
        :param stride: Base stride (dynamically handled within the constructor).
        """
        super(CNN, self).__init__()

        layers = []  # list to hold sequential layers
        channels = in_channels  # track input channels for each Conv2d

        for i in range(num_layers):
            out_channels = base_channels
            # apply stride only on the first layer to downsample once.
            # subsequent layers use (1, 1) to refine features at that resolution.
            stride = (2, 2) if i == 0 else (1, 1)

            # sequential block wrapping the core operations
            block = nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size, stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Dropout2d(p=0.1)
            )
            layers.append(block)
            channels = out_channels

        self.net = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass of the CNN feature extractor.

        What it does:
        - Iterates through the ModuleList.
        - Captures an identity (the input to the block).
        - Adds the identity back to the block output if the shapes match.

        Why:
        - Residual connections (x += identity) help the network learn identity
          mappings and prevent the vanishing gradient problem in deep stacks.
        - Padding=1 ensures that spatial dimensions are preserved in non-strided
          layers so residuals are mathematically possible.

        :param x: Input tensor of shape [Batch, Channel, Freq, Time].
        :return: High-level feature map tensor for the RNN.
        """
        for layer in self.net:
            identity = x
            x = layer(x)
            # residual addition. Only performed if stride was (1,1)
            # and input/output channels are the same.
            if x.shape == identity.shape:
                x += identity
        return x
