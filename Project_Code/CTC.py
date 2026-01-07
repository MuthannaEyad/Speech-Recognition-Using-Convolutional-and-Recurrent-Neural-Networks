"""
CTC Classifier Module

This file defines a simple fully connected layer followed by log softmax
used for Connectionist Temporal Classification (CTC) in speech recognition.

Design Choices:
- A single linear layer is sufficient because the feature extraction and sequence modelling
  are handled by the preceding CNN + RNN.

- Log softmax is applied along the class dimension to produce log-probabilities required
  by nn.CTCLoss, which expects log-probabilities rather than raw scores.
"""

import torch.nn as nn


class CTCClassifier(nn.Module):
    """
    Connectionist Temporal Classification (CTC) Classifier.

    What it does:
    - Receives a sequence of hidden states from the bidirectional RNN.
    - Projects these states into the dimension of the target alphabet (e.g., A-Z, space, blank).
    - Normalizes the output into log-probabilities.

    Why:
    - A multi-layer projection (Linear -> GELU -> Dropout -> Linear) is used to provide
      additional capacity for character discrimination after the temporal modeling.
    """

    def __init__(self, input_size, num_classes, dropout=0.1):
        """
        Initializes the CTC classifier architecture.

        What it does:
        - Sets up a sequential bottleneck architecture.
        - Uses GELU activation for smoother gradient flow compared to ReLU.
        - Implements dropout to prevent the classifier from over-relying on specific
          feature dimensions.

        Why:
        - Reducing the dimension (input_size // 2) before the final output acts as a
          regularizer, forcing the model to learn the most salient features for
          character classification.

        :param input_size: Dimensionality of RNN output at each time step.
        :param num_classes: Number of output classes (alphabet size + 1 for blank).
        :param dropout: Probability of dropout for regularization.
        """
        super(CTCClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 2, num_classes),
        )

    def forward(self, x):
        """
        Executes the forward pass to generate log-probabilities.

        What it does:
        - Maps the batch through the sequential layers.
        - Applies log_softmax across the class dimension (dim=2).

        Why:
        - Log-softmax is mathematically more stable than applying softmax followed by
          a log. CTCLoss specifically requires log-probabilities to avoid numerical
          underflow during the calculation of many possible alignments.

        :param x: RNN output tensor of shape [batch_size, time_steps, input_size].
        :return: Log-probabilities of shape [batch_size, time_steps, num_classes].
        """
        return self.classifier(x).log_softmax(dim=2)
