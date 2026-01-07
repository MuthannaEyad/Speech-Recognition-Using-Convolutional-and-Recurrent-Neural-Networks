"""
SpeechRecognizer Module for End-to-End Speech Recognition

This file defines the main speech recognition architecture combining:
1. CNN: Convolutional layers for feature extraction from log-Mel spectrograms.
2. Fully Connected Layer: Projects CNN features into the RNN input dimension.
3. RNN (GRU): Bidirectional recurrent layers to model temporal dependencies.
4. CTC Classifier: Linear + LogSoftmax layer for Connectionist Temporal Classification (CTC) loss training.

Design Choices:
- CNN extracts hierarchical spectral-temporal features from spectrograms.

- Fully connected layer adapts CNN output shape to the RNN hidden size.

- Bidirectional GRU allows the model to access both past and future context.

- CTC is used for alignment-free transcription, suitable for variable-length sequences.

- CNN output size is inferred dynamically to handle different input sequence lengths.

"""

import torch
import torch.nn as nn

# internal module imports
from CNN import CNN
from RNN import RNN
from CTC import CTCClassifier


class SpeechRecognizer(nn.Module):
    """
    End-to-End Speech Recognition Model.
    """

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, stride=2):
        """
        Initialize the SpeechRecognizer.

        :param n_cnn_layers: Number of CNN layers for feature extraction.
        :param n_rnn_layers: Number of RNN (GRU) layers.
        :param rnn_dim: Hidden dimension for RNN layers.
        :param n_class: Number of output classes for CTC (including blank).
        :param stride: Optional CNN stride for temporal subsampling.

        Variables:
        - self.cnn: CNN feature extractor.
        - self.fully_connected: Linear layer to project CNN features to RNN input.
        - self.rnn: Bidirectional GRU layers for temporal modelling.
        - self.ctc: Linear + LogSoftmax layer for CTC training.
        """
        super(SpeechRecognizer, self).__init__()

        self.stride = stride
        self.n_cnn_layers = n_cnn_layers
        self.n_rnn_layers = n_rnn_layers

        # CNN feature extractor
        # process raw spectrogram to find patterns
        self.cnn = CNN(
            num_layers=n_cnn_layers,
            in_channels=1,  # Log-Mel spectrograms are treated as 1-channel images
            base_channels=32,  # starting filter count
            kernel_size=3,  # standard 3x3 convolution
            stride=stride  # reduces the time dimension to speed up RNN processing
        )

        # we pass a dummy tensor through the CNN to calculate the exact output shape.
        # this prevents manual math errors when changing CNN layers or Mel bins.
        with torch.no_grad():
            # [Batch=1, Channel=1, Freq=128, Time=200]
            dummy_input = torch.zeros(1, 1, 128, 200)
            dummy_output = self.cnn(dummy_input)
            _, channels, freq, time = dummy_output.shape
            # the RNN expects a flat vector per timestep
            cnn_feature_dim = channels * freq

        # fully connected layer maps CNN features to RNN input dimension
        self.fully_connected = nn.Linear(cnn_feature_dim, rnn_dim)

        # bidirectional RNN for temporal modelling
        self.rnn = RNN(input_size=rnn_dim, hidden_size=rnn_dim, num_layers=n_rnn_layers)

        # CTC classifier has log-softmax outputs for CTC loss
        self.ctc = CTCClassifier(input_size=rnn_dim * 2, num_classes=n_class)

    def forward(self, x, input_length):
        """
        Forward pass through the SpeechRecognizer.

        :param x: Input tensor of shape [B, 1, F, T], where F=number of Mel bins.
        :param input_length: Original input lengths (not used for now, can be used for masking).
        :return:
            - x: Log probabilities [B, T', n_class] for CTC loss.
            - output_lengths: Lengths of CNN output sequences.

        Design Choices:
        - CNN extracts spatial-temporal features.
        - Flatten and permute to match RNN input [B, T, features].
        - Fully connected layer projects CNN features to RNN dimension.
        - Bidirectional RNN models temporal dependencies.
        - CTC classifier produces log-softmax probabilities for each timestep.
        """
        # CNN feature extraction
        x = self.cnn(x)

        # calculate the actual length as stride in CNN reduces the time-dimension
        output_lengths = input_length // self.stride

        # change CNN output from [B, C, F, T] to [B, T, C, F]
        x = torch.permute(x, (0, 3, 1, 2))

        # ensure output_lengths matches the actual tensor size to prevent CTC errors
        output_lengths = torch.clamp(output_lengths, max=x.size(1))
        output_lengths = output_lengths.to(x.device)

        x = torch.flatten(x, start_dim=2)  # flatten C*F per timestep

        # fully connected projection
        x = self.fully_connected(x)

        # bidirectional RNN optimizations for multi-GPU or newer hardware
        if hasattr(self.rnn, 'flatten_parameters'):
            self.rnn.flatten_parameters()

        x = self.rnn(x)

        # CTC classifier: output log probabilities
        x = self.ctc(x)

        return x, output_lengths
