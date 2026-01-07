"""
Dataset Preprocessing for LibriSpeech

This file handles the creation of processed datasets for speech recognition training
and testing.

The preprocessing includes:
1. Resampling all audio to a consistent 16kHz sample rate.
2. Converting audio waveforms into log-mel spectrogram's.
3. Encoding transcripts into integer sequences for CTC-based training.
4. Saving each processed sample as a `.pt` file for faster data loading.

Design Choices:
- MelSpectrogram parameters (n_mels=80, n_fft=400, hop_length=160):
  Standard choice in ASR for capturing sufficient frequency resolution while
  keeping temporal resolution manageable.

- Uppercasing transcripts and limiting vocabulary to A-Z, space, and blank:
  Simplifies decoding and aligns with standard English speech datasets.

- Saving each sample individually:
  Allows random access and efficient loading in PyTorch DataLoader with
  custom collate functions.

- `collate` function pads variable-length sequences for batch processing
  compatible with CTC loss.

WARNING:
This script consumes significant memory from the heap.
Processed datasets are already provided in the `./dataset/processed` folder.
"""
import os
import gc
import glob
import time
import torch
import string
import soundfile as sf
import torchaudio.transforms as T

# root directories for raw and processed data
DATA_ROOT = "./dataset"
OUT_ROOT = "./dataset/processed"


# unfortunately the below method for preprocessing the data
# preprocess_split cannot be used as it used too much memory instead the
# preprocess_manually function was designed to try and use as little memory as possible at the cost of time to
# fully process
def preprocess_split(dataset, out_dir, char2index):
    """
    Preprocesses a LibriSpeech dataset split and saves each sample as a .pt file.

    :param dataset: Torchaudio LibriSpeech dataset object.
    :param out_dir: Directory to save processed samples.
    :param char2index: Dictionary mapping characters to integer indices.
    :return: None

    Variables:
    - mel: MelSpectrogram transform to convert waveform to mel-spectrogram.
    - db: Converts amplitude to decibels.
    - waveform: Raw audio tensor for each sample.
    - features: Log-mel spectrogram tensor [1, n_mels, T].
    - targets: Encoded transcript as a list of integers.
    - sample: Dictionary containing features, targets, and lengths for saving.
    """
    mel = T.MelSpectrogram(
        sample_rate=16000,
        n_mels=128,
        n_fft=512,
        hop_length=160
    )
    db = T.AmplitudeToDB()

    resampler_48k = T.Resample(orig_freq=48000, new_freq=16000)

    for index, (waveform, sample_rate, transcript, *_) in enumerate(dataset):
        # standardize all audio to 16kHz for consistent feature extraction
        # waveform = resample_audio(waveform, sample_rate)
        if sample_rate == 48000:
            waveform = resampler_48k(waveform)
        elif sample_rate != 16000:
            # Fallback for rare cases of different rates
            tmp_resample = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = tmp_resample(waveform)
        # convert waveform to log-mel spectrogram
        features = db(mel(waveform))

        features = (features - features.mean()) / (features.std() + 1e-7)

        # encode transcript into integer indices
        targets = encode_transcript(transcript, char2index)
        if len(targets) == 0:
            continue

        # sample dictionary contains everything needed for training
        sample = {
            "features": features,  # log-mel spectrogram [1, n_mels, T]
            "targets": torch.tensor(targets, dtype=torch.long),
            "input_length": features.shape[-1],  # length of time dimension
            "target_length": len(targets),  # length of transcript
        }

        # save individual sample for efficient loading later
        torch.save(sample, f"{out_dir}/{index:06d}.pt")

        if index % 200 == 0:
            os.sync()
            gc.collect()
            time.sleep(2)


def preprocess_manually(split_name, char2index):
    """
    Manually iterates through the LibriSpeech directory to process audio files one by one.

    This is to prevent bad allocation errors.

    :param split_name: Name of the dataset split (e.g., 'train-clean-100').
    :param char2index: Dictionary mapping characters to integer indices.
    """

    mel_transform = T.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=160)
    db_transform = T.AmplitudeToDB()

    raw_path = os.path.join(DATA_ROOT, "LibriSpeech", split_name, "**/*.flac")
    files = glob.iglob(raw_path, recursive=True)

    out_dir = os.path.join(OUT_ROOT, split_name)
    os.makedirs(out_dir, exist_ok=True)

    for index, file_path in enumerate(files):
        out_file = f"{out_dir}/{index:06d}.pt"
        if os.path.exists(out_file): continue

        try:
            # use SoundFile instead of torchaudio to bypass C++ backend crashes.
            speech, sample_rate = sf.read(file_path)
            waveform = torch.from_numpy(speech).unsqueeze(0).float()

            with torch.no_grad():
                if sample_rate != 16000:
                    waveform = T.Resample(sample_rate, 16000)(waveform)

                features = db_transform(mel_transform(waveform))
                features = (features - features.mean()) / (features.std() + 1e-7)

                raw_text = get_transcript(file_path)
                targets = encode_transcript(raw_text, char2index)

                if len(targets) > 0:
                    # saving as Half-Precision to reduce disk I/O bottlenecks.
                    torch.save({
                        "features": features.half(),  # Float16 for HDD
                        "targets": torch.tensor(targets, dtype=torch.long),
                        "input_length": features.shape[-1],
                        "target_length": len(targets),
                    }, out_file)

            # manual memory purge
            del waveform, speech, features

        except Exception as e:
            print(f"Skipped {index} due to error: {e}")

        if index % 5 == 0:
            gc.collect()
            os.sync()


def get_transcript(file_path):
    """
    Locates and parses the transcript text for a specific audio file.

    :param file_path: Path to the raw .flac file.
    :return: Cleaned transcript string.
    """
    dir_path = os.path.dirname(file_path)
    file_id = os.path.basename(file_path).replace(".flac", "")

    trans_files = glob.glob(os.path.join(dir_path, "*.trans.txt"))
    if not trans_files:
        return ""

    with open(trans_files[0], "r") as f:
        for line in f:
            if line.startswith(file_id):
                return line.strip().replace(file_id, "").strip()
    return ""


def resample_audio(waveform, sample_rate):
    """
    Resamples input audio to 16kHz if necessary.

    :param waveform: Tensor of shape [channels, samples].
    :param sample_rate: Original sampling rate.
    :return: Resampled waveform tensor.

    Design Choice:
    - All audio is resampled to 16kHz to maintain consistent temporal resolution
      for MelSpectrogram features.
    """
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform


def build_char_vocab():
    """
    Builds character-level vocabulary for ASR tasks.

    :return: char2index, index2char dictionaries

    Design Choice:
    Vocabulary includes:
    - <blank> token for CTC loss.
    - Space character.
    - Uppercase English letters A-Z.
    """
    vocab = ["<blank>", " "] + list(string.ascii_uppercase)

    char2index = {ch: i for i, ch in enumerate(vocab)}
    index2char = {i: ch for i, ch in enumerate(vocab)}

    return char2index, index2char


def encode_transcript(text, char2index):
    """
    Converts a transcript string to a list of integer indices.

    :param text: Raw transcript string.
    :param char2index: Character-to-index mapping.
    :return: List of integer indices corresponding to characters.

    Design Choice:
     - Uppercase all letters to match vocabulary.
     - Ignore characters not in vocabulary.
    """
    text = text.upper()
    return [char2index[c] for c in text if c in char2index]


def decode_transcript(indices, index2char):
    """
    Decodes a predicted index sequence into a string.

    Collapses repeated characters and removes blank tokens.

    :param indices: List of predicted indices.
    :param index2char: Index-to-character mapping.
    :return: Decoded string.
    """
    prev = None
    output = []

    for index in indices:
        if index != prev and index != 0:  # remove repeats and blank token
            output.append(index2char[index])
        prev = index

    return "".join(output)


def decode_raw(indices, index2char):
    """
    Decodes a raw sequence of indices into string, ignoring blank tokens.

    :param indices: List of indices.
    :param index2char: Index-to-character mapping.
    :return: Decoded string.
    """
    return "".join(index2char[i] for i in indices if i != 0)


def collate(batch):
    """
    Custom collate function for DataLoader.

    Pads variable-length feature sequences and concatenates targets for batch
    processing in CTC training.

    :param batch: List of tuples (features, targets, input_length, target_length)
    :return:
            Padded features [B, 1, n_mels, T_max].
            Concatenated targets.
            Tensor of input_lengths.
            Tensor of target_lengths.

    Variables:
    - features: Padded mel-spectrogram features.
    - targets: Concatenated target sequences.
    - input_length: Original lengths of each feature sequence.
    - target_length: Original lengths of each target sequence.
    """
    features, targets, input_length, target_length = zip(*batch)

    # pad variable-length sequences along time dimension
    features = torch.nn.utils.rnn.pad_sequence(
        [f.squeeze(0).transpose(0, 1) for f in features],
        batch_first=True,
        padding_value=0.0
    ).transpose(1, 2).unsqueeze(1)  # shape: [B, 1, n_mels, T_max]

    targets = torch.cat(targets).long()

    input_length = torch.tensor(input_length, dtype=torch.long)
    target_length = torch.tensor(target_length, dtype=torch.long)

    return features, targets, input_length, target_length


if __name__ == "__main__":
    # make the maps for char to index and vice versa and same it in a single vocab file
    char2index, index2char = build_char_vocab()
    os.makedirs(OUT_ROOT, exist_ok=True)
    torch.save({"char2index": char2index, "index2char": index2char}, f"{OUT_ROOT}/vocab.pt")

    # preprocess both training and testing sets
    preprocess_manually("train-clean-100", char2index)
    preprocess_manually("test-clean", char2index)
