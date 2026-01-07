"""
Preprocessed Speech Dataset Loader

This file defines a PyTorch Dataset class to load preprocessed LibriSpeech data
saved as individual `.pt` files. Each sample contains:

- features: Log-mel spectrogram tensor [1, 80, T].
- targets: Encoded transcript as tensor [L].
- input_length: Length of the feature sequence (time dimension).
- target_length: Length of the target sequence (number of characters).

Design Choices:
- Each sample is saved individually for efficient random access during training.

- Sorting file names ensures consistent ordering across runs.

- Returns raw tensors; padding and batching are handled in the collate function.
"""

import os
import torch
from torch.utils.data import Dataset


class PreprocessedSpeechDataset(Dataset):
    """
    PyTorch Dataset for preprocessed speech data.

    What it does:
    Acts as the interface between the saved .pt files on disk and the training loop.
    It indexes all available processed samples and provides them on-demand.

    Why:
    Using a Dataset class allows us to leverage PyTorch's DataLoader for
    parallelized data loading (num_workers), ensuring that the GPU is never
    waiting for the CPU to fetch the next batch.
    """

    def __init__(self, data_dir):
        """
        Initializes the dataset by collecting all file paths in the directory.

        What it does:
        - Scans the directory for .pt files.
        - Stores a sorted list of absolute paths.

        Why:
        - Sorting is critical for deterministic behavior; it ensures that
          shuffling behaves predictably across different training sessions.
        - Storing paths instead of loading data here prevents a "Memory Bloat"
          at the start of the script.

        :param data_dir: Directory containing preprocessed .pt files.
        """
        # list all files in the directory and join with path
        self.files = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
        )

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        :return: Number of .pt files.
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Loads and returns a single sample from the dataset.

        What it does:
        - Loads a serialized dictionary from disk using torch.load.
        - Explicitly converts half-precision features back to float32.
        - Returns the features, targets, and their respective original lengths.

        Why:
        - map_location='cpu' and weights_only=True are used as a safety layer
          to prevent VRAM spikes during the initial load.
        - Converting back to float32 is essential because while we save in
          float16 to save disk space, most neural network layers and
          ROCm/CUDA kernels require float32 for high-precision math.

        :param idx: Index of the sample to load.
        :return: Tuple containing (features, targets, input_length, target_length).
        """
        # load the dictionary from the disk
        sample = torch.load(self.files[idx], map_location='cpu', weights_only=True)

        # convert stored half-precision back to float32 for model compatibility
        features = sample["features"].float()
        targets = sample["targets"].long()

        return (
            features,  # [1, 128, T]
            targets,  # [L]
            sample["input_length"],  # T
            sample["target_length"]  # L
        )
