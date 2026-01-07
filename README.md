# Automatic Speech Recognition with CTC (PyTorch)

This project implements a character-level automatic speech recognition (ASR) system trained on LibriSpeech using CTC-based alignment-free learning.

## Key Features
- CNN–BiGRU architecture for end-to-end speech recognition
- Log-Mel spectrogram preprocessing pipeline
- CTC loss with greedy decoding (no external language model)
- Mixed-precision training and checkpointed training pipeline

## Dataset
- LibriSpeech (train-clean-100, test-clean)
- ~7GB of raw audio (not stored in repository)

## Results
- Final validation WER ≈ 0.26 on clean speech
- No external language model used

## Why this project
Built to understand sequence modeling, alignment-free training, and large-scale audio pipelines from first principles.

## Tech Stack
PyTorch, CUDA, LibriSpeech, CNN, BiGRU, CTC
