import os
os.environ["MIOPEN_FIND_MODE"] = "FAST"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
from jiwer import wer
from torch.utils.data import DataLoader

import PreprocessedSpeechDataset as psd
from SpeechRecognition import SpeechRecognizer
from DataProcessing import collate, decode_transcript, decode_raw

TEST_DIR = "../dataset/processed/test-clean"
VOCAB_DIR = "../dataset/processed/vocab.pt"
CHECKPOINT_DIR = "../checkpoints/"
FINAL_MODEL_PATH = "./final_model.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODELS_BEING_TESTED = [
    "../checkpoints/checkpoint_epoch2.pt",
    # "../checkpoints/checkpoint_epoch4.pt",
    # "../checkpoints/checkpoint_epoch5.pt",
    # "../checkpoints/checkpoint_epoch6.pt",
    # "../checkpoints/checkpoint_epoch8.pt",
    "../checkpoints/checkpoint_epoch11.pt",
    # "../checkpoints/checkpoint_epoch16.pt",
    # "../checkpoints/checkpoint_epoch40.pt",
    # "../checkpoints/checkpoint_epoch50.pt",
    # "../checkpoints/checkpoint_epoch55.pt",
    # "../checkpoints/checkpoint_epoch60.pt",
    FINAL_MODEL_PATH
]

CHOSEN_SAMPLES = []
NUM_SAMPLES = 3


def check_prediction(pre_select_samples=False):
    vocab = torch.load(VOCAB_DIR)
    index2char = vocab["index2char"]
    testing_set = psd.PreprocessedSpeechDataset(TEST_DIR)

    samples = []
    if pre_select_samples:
        for i in CHOSEN_SAMPLES:
            raw_sample = testing_set[i]
            samples.append(collate([raw_sample]))
    else:
        test_loader = DataLoader(testing_set, batch_size=1, collate_fn=collate, shuffle=False)
        data_itr = iter(test_loader)
        for _ in range(NUM_SAMPLES):
            samples.append(next(data_itr))

    model = SpeechRecognizer(
        n_cnn_layers=3, n_rnn_layers=3, rnn_dim=512, n_class=28, stride=2
    ).to(DEVICE)

    for checkpoint_path in MODELS_BEING_TESTED:
        if not os.path.exists(checkpoint_path):
            print(f"--- Checkpoint {checkpoint_path} not found, skipping ---")
            continue

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        # Handle difference between checkpoint dict and direct state_dict
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        print(f"\n{'=' * 25} {os.path.basename(checkpoint_path).upper()} {'=' * 25}")

        with torch.no_grad():
            for i, (features, targets, input_lengths, target_lengths) in enumerate(samples):
                features, input_lengths = features.to(DEVICE), input_lengths.to(DEVICE)

                log_probs, output_lengths = model(features, input_lengths)
                log_probs = log_probs.transpose(0, 1)

                arg_maxes = torch.argmax(log_probs, dim=2)
                pred_indices = arg_maxes[:, 0][:output_lengths[0]].tolist()
                prediction = decode_transcript(pred_indices, index2char)

                target_str = decode_raw(targets.tolist(), index2char)

                print(f"SAMPLE {i + 1}")
                print(f"  TARGET  : {target_str}")
                print(f"  PRED: {prediction}")
                print(f"  WER : {wer(target_str, prediction):.4f}")


if __name__ == "__main__":
    check_prediction(pre_select_samples=False)
