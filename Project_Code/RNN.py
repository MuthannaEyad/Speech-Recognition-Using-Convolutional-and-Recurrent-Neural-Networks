"""
RNN Module for Speech Recognition

This file defines a recurrent neural network (RNN) using a bidirectional GRU.
The RNN is responsible for modelling temporal dependencies in the feature sequences
produced by the preceding CNN feature extractor.

Design Choices:
- GRU is used over vanilla RNNs for better long-term dependency learning.

- Bidirectional GRU is used to capture both past and future context, which
  improves performance in speech recognition tasks.

- batch_first=True allows inputs in shape [batch_size, time_steps, features],
  consistent with the output of our preprocessing pipeline and DataLoader.
"""

import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    """
    Recurrent Neural Network (Bidirectional GRU) for sequence modelling.

    What it does:
    - Processes sequential audio features through multiple layers of bidirectional GRUs.
    - Applies Layer Normalization and GELU activation after each recurrent pass.
    - Utilizes residual additions where input and output dimensions match.

    Why:
    - Stacked GRUs allow the model to learn increasingly complex temporal hierarchies.
    - LayerNorm and Residuals prevent the 'vanishing gradient' problem, which is common
       in deep recurrent architectures.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        """
        Initializes the stacked RNN architecture.

        What it does:
        - Creates a ModuleList of GRU layers and corresponding LayerNorm modules.
        - Calculates the input size for each layer, doubling the hidden_size for
          subsequent layers to account for bidirectional concatenation.

        Why:
        - Using nn.ModuleList instead of a single multi-layer nn.GRU allows us
          to insert normalization and residual logic between each specific layer.

        :param input_size: Dimensionality of input features (from CNN or FC layer).
        :param hidden_size: Number of features in the hidden state of each GRU direction.
        :param num_layers: Number of stacked bidirectional layers.
        :param dropout: Dropout probability for temporal regularization.
        """
        super(RNN, self).__init__()
        self.rnns = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            # the first layer takes CNN features; subsequent layers take the
            # concatenated [forward, backward] output (hidden_size * 2) of the previous layer.
            layer_input_size = input_size if i == 0 else hidden_size * 2

            self.rnns.append(
                nn.GRU(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_size * 2))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the RNN.

        What it does:
        - Iteratively passes the data through each GRU, Normalization, and Activation block.
        - Adds the residual (input) back to the output if the dimensions are identical.

        Why:
        - GELU (Gaussian Error Linear Unit) provides a smoother non-linearity than ReLU,
          often leading to better convergence in transformer and ASR architectures.
        - Bidirectionality doubles the feature dimension (hidden_size * 2) because
          it concatenates the hidden states from the forward and backward passes.

        :param x: Input tensor of shape [batch_size, time_steps, input_size].
        :return: Output tensor of shape [batch_size, time_steps, hidden_size*2].
        """
        for i in range(len(self.rnns)):
            residual = x

            x, _ = self.rnns[i](x)

            x = self.norms[i](x)
            x = F.gelu(x)

            if x.shape == residual.shape:
                x += residual

            x = self.dropout(x)

        return x
