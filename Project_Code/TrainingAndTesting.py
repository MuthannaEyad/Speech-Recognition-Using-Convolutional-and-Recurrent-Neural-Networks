"""
Training and Evaluation Script for Speech Recognition

This file contains all the logic required to train, evaluate, and track
performance for the SpeechRecognizer model using the preprocessed
LibriSpeech dataset.

Features:
- Training loop with CTC loss
- Evaluation loop with Word Error Rate (WER) tracking
- Checkpoint saving per epoch and final model saving
- Optional resume training from a checkpoint
- Decoding functions for CTC outputs and raw target sequences
- Compatibility layer for both AMD (ROCm) and NVIDIA (CUDA) GPUS
"""
import os

# tells MIOpen to use a faster heuristic to find the best convolution algorithms instead of
# exhaustive searching
os.environ["MIOPEN_FIND_MODE"] = "FAST"

# tricks the ROCm driver into treating your GPU as a compatible GFX1030 (RX 6000 series) device
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

# prevents VRAM fragmentation by allowing PyTorch to use more flexible,
# expandable memory segments
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# defines a local path to store pre-compiled GPU kernels so they don't have to be rebuilt
# every run
cache_dir = os.path.join(os.getcwd(), "miopen_cache")

# ensures the physical directory for the GPU kernel cache exists on your drive
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# silences MIOpen's verbose output, showing only critical errors (Level 3)
os.environ["MIOPEN_LOG_LEVEL"] = "3"

# pins MIOpen operations to the first available GPU in the system (ID 0)
os.environ["MIOPEN_DEVICE_UUID"] = "0"

# points MIOpen to the specific database file location for your cached kernels.
os.environ["MIOPEN_USER_DB_PATH"] = cache_dir

# points MIOpen to the directory where it should store custom-compiled kernel binaries.
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = cache_dir

# extends the allowed wait time for a kernel to compile before the system throws a timeout error.
os.environ["MIOPEN_DEBUG_COMPILED_KERNEL_TIMEOUT"] = "10"


import csv
import time
import torch
import numpy as np
import torch.nn as nn

from jiwer import wer
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# custom imports for model architecture, dataset class, and data collation
import PreprocessedSpeechDataset as psd
from SpeechRecognition import SpeechRecognizer
from DataProcessing import collate, decode_transcript, decode_raw


# ====================== HARDWARE SETUP ======================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# benchmark is set False to ensure deterministic/stable behavior on different GPUs
torch.backends.cudnn.benchmark = False

# ====================== CONFIGURATION ======================

# directory paths for data and checkpoints
TRAIN_DIR = "../dataset/processed/train-clean-100"
TEST_DIR = "../dataset/processed/test-clean"
VOCAB_DIR = "../dataset/processed/vocab.pt"
CHECKPOINT_DIR = "../checkpoints/"
FINAL_MODEL_PATH = "../checkpoints/final_model.pt"

# ensure checkpoint directory exists if not it creates one
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# hyperparameters
BATCH_SIZE = 32  # number of samples per batch
ACCUMULATION_STEPS = 4  # Simulates a batch size of 128 (32 * 4) to improve gradient stability
LEARNING_RATE = 1e-3
EPOCHS = 1000  # maximum number of training iterations

# load character vocab mapping
vocab = torch.load(VOCAB_DIR)
char2index = vocab["char2index"]  # maps character to integer index
index2char = vocab["index2char"]  # maps integer index back to character

# ====================== DATA LOADING ======================

# dataset instances using the preprocessed LibriSpeech data
training_set = psd.PreprocessedSpeechDataset(TRAIN_DIR)
testing_set = psd.PreprocessedSpeechDataset(TEST_DIR)

# generate indices for batching
indices_train = np.arange(len(training_set))
indices_test = np.arange(len(testing_set))

# pre-group indices into batches allowing for consistent batch sampling
batches_train = [indices_train[i:i + BATCH_SIZE] for i in range(0, len(indices_train), BATCH_SIZE)]
batches_test = [indices_test[i:i + BATCH_SIZE] for i in range(0, len(indices_test), BATCH_SIZE)]

# validation loader (shuffling is False to keep snapshots consistent)
test_loader = DataLoader(
    testing_set,
    batch_sampler=batches_test,
    collate_fn=collate,
    num_workers=8,  # reduce this value if cpu usage is too high or crashes
    pin_memory=True,  # accelerates memory transfer form CPU to GPU
    persistent_workers=True,
    prefetch_factor=2  # reduce this for same reason as num_workers
)

# ====================== TRAINING LOOP ======================

scaler = GradScaler()  # handles mixed precision training (FP16) to prevent underflow


def train_one_epoch(model, optimizer, criterion, loader, epoch_index):
    """
    Train the model for one epoch using CTC Loss and Gradient Accumulation.

    :param model: SpeechRecognizer instance.
    :param optimizer: Optimizer instance (Adam).
    :param criterion: CTC loss function.
    :param loader: DataLoader for training data.
    :param epoch_index: Current epoch number.
    :return: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    # progress bar to show how many batches have been completed as well as how much time until
    # a single epoch is completed
    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch_index + 1}/{EPOCHS}",
        unit="batch",
        leave=False
    )

    for batch_idx, (features, targets, input_lengths, target_lengths) in enumerate(progress_bar):
        # move data to GPU (non_blocking=True improves overlap between CPU and GPU)
        features = features.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        input_lengths = input_lengths.to(DEVICE, non_blocking=True)
        target_lengths = target_lengths.to(DEVICE, non_blocking=True)

        # forward pass with mixed precision (autocast)
        # autocast allows certain operations to run in Float16, doubling
        # training speed and reducing VRAM usage
        with autocast(device_type="cuda"):
            log_probs, output_lengths = model(features, input_lengths)
            log_probs = log_probs.transpose(0, 1)  # CTC expects [time, batch, class]

            loss = criterion(log_probs, targets, output_lengths, target_lengths)
            # normalized loss for accumulation steps
            loss = loss / ACCUMULATION_STEPS

        # scale gradients and backpropagate
        scaler.scale(loss).backward()

        # the commented out if statement below is to check if the target and output lengths match
        # if batch_idx == 0:
        #     print(f"DEBUG: Input Max Length: {input_lengths.max().item()}")
        #     print(f"DEBUG: CNN Output T: {log_probs.size(0)}")  # T is first after transpose

        # a check to see if the CNN didn't over compress the audio too much
        if (output_lengths < target_lengths).any():
            print("WARNING: CNN subsampling is too aggressive! Target longer than Output.")
            print(f"Target Length: {target_lengths}, Output Length: {output_lengths}")

        # update model weights every ACCUMULATION_STEPS
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            # clipped gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # clean memory specifically for AMD/ROCm stability
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        total_loss += loss.item() * ACCUMULATION_STEPS
        progress_bar.set_postfix(loss=total_loss / (batch_idx + 1))

    return total_loss / len(loader)


# ====================== EVALUATION LOOP ======================

def evaluate(model, criterion, loader):
    """
    Evaluate the model and compute Word Error Rate (WER).

    :param model: SpeechRecognizer instance.
    :param criterion: CTC loss function.
    :param loader: DataLoader for validation/test data.
    :return: Tuple (average loss, average WER).
    """
    model.eval()
    total_loss, total_wer, total_samples = 0.0, 0.0, 0

    progress_bar = tqdm(loader, desc="Evaluating", unit="batch", leave=False)

    with torch.no_grad():  # disable gradient calculation to save memory and speed up inference
        for i, (features, targets, input_lengths, target_lengths) in enumerate(progress_bar):
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            input_lengths, target_lengths = input_lengths.to(DEVICE), target_lengths.to(DEVICE)

            log_probs, output_lengths = model(features, input_lengths)
            log_probs = log_probs.transpose(0, 1)

            loss = criterion(log_probs, targets, output_lengths, target_lengths)
            total_loss += loss.item()

            # decode predictions and ground truths
            pred_texts = greedy_decode(log_probs, output_lengths, index2char)
            target_texts = decode_targets(targets, target_lengths, index2char)

            # printing three samples to track evolution
            if i == 0:
                # Samples 0, 10, and 20 usually have different lengths/complexities
                for idx in [0, 10, 20]:
                    print(f"\nSAMPLE {idx}")
                    print(f"  TARGET: {target_texts[idx]}")
                    print(f"  PRED:   '{pred_texts[idx]}'\n")

            for pred, target in zip(pred_texts, target_texts):
                total_wer += wer(target, pred)
                total_samples += 1

    avg_loss = total_loss / len(loader)
    avg_wer = total_wer / total_samples

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return avg_loss, avg_wer


# ====================== HELPER FUNCTIONS ======================

def greedy_decode(log_probs, output_lengths, index2char):
    """
    Greedy decoding of CTC log probabilities. Collapses repeating characters
    and removes the blank token (0)

    :param log_probs: Log probabilities [T, B, C], [B, T, C].
    :param output_lengths: Original output lengths from the model.
    :param index2char: Dictionary mapping index to character.
    :return: List of decoded strings.
    """
    # standardize shape to [batch, time class]
    if log_probs.size(1) == len(output_lengths):
        predictions = log_probs.argmax(dim=2).transpose(0, 1)
    else:
        predictions = log_probs.argmax(dim=2)

    decoded = []
    for i in range(predictions.size(0)):
        length = output_lengths[i].item()
        sequence = predictions[i][:length].tolist()
        decoded.append(decode_transcript(sequence, index2char))
    return decoded


def decode_targets(targets, target_lengths, index2char):
    """
    Converts raw target integer sequences back into readable strings.

    :param targets: Flattened target tensors.
    :param target_lengths: List of lengths for each sample in the batch.
    :param index2char: Dictionary mapping index to char.
    :return: List of decoded target strings.
    """
    decoded = []
    offset = 0
    for length in target_lengths:
        sequence = targets[offset:offset + length].tolist()
        decoded.append(decode_raw(sequence, index2char))
        offset += length
    return decoded


# ====================== MAIN EXECUTION ======================

def main(resume_training=False, training_file=""):
    """
    Main entry point for training and evaluation.

    :param resume_training: If True, resume from a checkpoint.
    :param training_file:  Filename to resume from.
    """
    if resume_training and (training_file.strip() == ""):
        raise ValueError("You must specify a training file")

    # initialize model architecture
    model = SpeechRecognizer(
        n_cnn_layers=3,
        n_rnn_layers=3,
        rnn_dim=512,
        n_class=28,  # blank + space + A-Z
        stride=2
    ).to(DEVICE)

    # initialize loss, optimizer, and learning rate scheduler
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    # reduces lr when validation loss stops decreasing to fine-tune weights
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    # handle checkpoint loading if resuming
    start_epoch = 0
    if resume_training:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, training_file)
        if os.path.exists(checkpoint_path):
            # map_location ensures GPU saves can be loaded onto different GPUs/CPUs
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])

            # if training has plateaued uncomment below and make learning rate bigger
            # for param_group in optimizer.param_groups:
            #    param_group["lr"] = LEARNING_RATE

            start_epoch = checkpoint["epoch"] + 1
            if "scheduler_state" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            print(f"Resuming training from epoch {start_epoch}")

    # set up data loader for training
    train_loader = DataLoader(
        training_set,
        batch_sampler=batches_train,
        collate_fn=collate,  # pads variable-length features, concatenates targets
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # set up file for metrics
    LOG_FILE = "./training_metrics.csv"
    if not resume_training:
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_wer", "epoch_time"])

    # training state variables
    patience = 15
    epochs_without_improvement = 0
    best_val_loss = float("inf")
    total_start_time = time.perf_counter()  # used to check time taken to fully train

    # training loop
    for epoch in range(start_epoch, EPOCHS):
        epoch_start_time = time.perf_counter()  # used to check the time taken per epoch

        # shuffle training batches every epoch
        np.random.shuffle(batches_train)

        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, epoch)
        val_loss, val_wer = evaluate(model, criterion, test_loader)

        # adjust learning rate based on validation performance
        scheduler.step(val_loss)

        # early stopping for when model stops improving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

        # save checkpoint for this epoch
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "scheduler_state": scheduler.state_dict()
        }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch + 1}.pt"))

        epoch_end_time = time.perf_counter()
        duration = epoch_end_time - epoch_start_time

        # enter metrics into the CSV
        with open(LOG_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, val_wer, duration])

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val WER: {val_wer:.4f} | "
            f"Epoch time: {duration:.2f}s"
        )

    # save final model
    torch.save(model.state_dict(), FINAL_MODEL_PATH)

    total_end_time = time.perf_counter()

    print(f"Total training time: {(total_end_time - total_start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    # to start fresh, set resume_training=False
    main(resume_training=True, training_file="checkpoint_epoch45.pt")
