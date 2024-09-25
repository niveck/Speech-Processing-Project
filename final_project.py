import os
import random
import torch
import torchaudio as ta
import torchaudio.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
# from transformers import pipeline
# from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
import numpy as np



RANDOM_SEED = 42

EPSILON = "_"
DIGITS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero"]
DIGITS_ALPHABET = ["e", "f", "g", "h", "i", "n", "o", "r", "s", "t", "u", "v", "w", "x", "z"]

INDEX2LETTER = dict(enumerate([EPSILON] + DIGITS_ALPHABET))  # original, with regular words
# we are starting the indices from 1 so we can negate them to signify masking the labels
LETTER2INDEX = {letter: index for index, letter in INDEX2LETTER.items()}

MAX_DIGIT_NAME_LENGTH = max([len(digit_name) for digit_name in DIGITS])  # 5
MAX_TIME_STEPS = 98  # in this specific dataset, used to save time in following trainings

# hyperparameters
SAMPLE_RATE = 16000
N_MFCC = 13
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 11  # 20
CONV_KERNEL_SIZE = 5
PADDING = 2
HIDDEN_DIM = 512
MEL_KWARGS = {"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False,
              "normalized": False  # True
              }
CHECKPOINT_INTERVAL = 3
LOGS_DIR = "results/180924halfdata"

# TRAIN_SET_PATH = "../ex3/data/train"
TRAIN_SET_PATH = "/cs/snapless/gabis/nive/speech/train"
# VAL_SET_PATH = "../ex3/data/val"
VAL_SET_PATH = "/cs/snapless/gabis/nive/speech/val"
# TEST_SET_PATH = "../ex3/data/test"
TEST_SET_PATH = "/cs/snapless/gabis/nive/speech/test"

# Training with LLM constants
TRAIN_DATA_WITHOUT_LLM_PERCENTAGE = 60
USE_LLM_FOR_LABELS = False
DONT_USE_LABELS_RATIO = 2  # once in how many samples we want to ignore label
NUM_CLEAN_EPOCHS = 5  # TODO consider playing with this hyperparameter
LLM_NAME = "gpt-3.5-turbo-0125"  # "gpt-j-6B"  # "gpt2"
# LLM_PROMPT = "We're training a CTC speech to text model and we decoded the prob matrix to get " \
#              "the word: '{decoded_output}'. Tell us what do you think the word was. Give us the " \
#              "answer with one word only (very important to only use one word, because I don't " \
#              "want your answer to be more than one word): "  # REMEMBER to use {decoded_output}
LLM_PROMPT = "My Speech-to-Text model is in the middle of training, and predicted the following" \
             " text for a recording of a digit name: '{decoded_output}'. Which digit name is the " \
             "most likely the true label of this recording? Reply with only one word out of the " \
             "following: zero, one, two, three, four, five, six, seven, eight, nine (remember " \
             "to only output one word out of them)."

# use boolean flag to only run with the half "tagged" data with no LLM help
REFERENCE_RUN_WITH_HALF_DATA = True


class DigitDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.digit_names = [digit_name for digit_name in os.listdir(root_dir)
                            if digit_name in DIGITS]
        self.file_names = {digit_name: [] for digit_name in self.digit_names}
        for digit_name in self.digit_names:
            digit_data_dir = os.path.join(root_dir, digit_name)
            # for file_name in os.listdir(digit_data_dir):
            for file_name in os.listdir(digit_data_dir):
                if file_name.endswith(".wav"):
                    self.file_names[digit_name].append(os.path.join(digit_data_dir, file_name))
        self.max_time_steps = MAX_TIME_STEPS + PADDING  # self.get_max_time_steps()
        self.max_label_length = MAX_DIGIT_NAME_LENGTH
        self.to_mfcc = transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, melkwargs=MEL_KWARGS)

    def __len__(self):
        return sum([len(audio_files) for digit, audio_files in self.file_names.items()])

    def __getitem__(self, unused_index):
        sampled_digits = random.sample(self.digit_names, 1)
        label_str = "".join([digit for digit in sampled_digits])  # original was w/o DIGITS2PHONEMES
        waveform = torch.cat([ta.load(random.choice(self.file_names[digit]))[0]
                              for digit in sampled_digits], dim=1)
        mfcc = self.to_mfcc(waveform)
        mfcc = torch.mean(mfcc, dim=0).transpose(0, 1)  # (time, mfcc_coefficients)
        return mfcc, label_str

    def get_max_time_steps(self):
        max_time_steps = 0
        for digit, file_list in self.file_names.items():
            for file_name in file_list:
                wav, _ = ta.load(file_name)
                mfcc = self.to_mfcc(wav)
                mfcc = torch.mean(mfcc, dim=0).transpose(0, 1)
                max_time_steps = max(max_time_steps, mfcc.shape[0])
        return max_time_steps + PADDING


def pad_tensors_and_vectorize_labels(batch, dataset):  # the collate function
    mfccs, labels = zip(*batch)
    padded_mfccs = torch.zeros(len(mfccs), dataset.max_time_steps, mfccs[0].shape[1])
    for i, mfcc in enumerate(mfccs):
        padded_mfccs[i, :mfcc.shape[0], :] = mfcc
    input_lengths = torch.tensor([mfcc.shape[0] for mfcc in mfccs])
    target_lengths = torch.tensor([len(label) for label in labels])
    padded_targets = labels_to_padded_targets(labels)
    return padded_mfccs, padded_targets, input_lengths, target_lengths


def labels_to_padded_targets(labels):
    labels = [[LETTER2INDEX[char] for char in label] for label in labels]
    padded_targets = torch.ones(len(labels), MAX_DIGIT_NAME_LENGTH) * LETTER2INDEX[EPSILON]
    for i, label in enumerate(labels):
        padded_targets[i, :len(label)] = torch.tensor(label)
    return padded_targets


def decode_output(logits):  # greedily (without beam search)
    batch_predictions = logits.argmax(dim=-1).t()
    decoded_sequences = []
    for sequence in batch_predictions:
        decoded_sequence = decode_single_sequence(sequence)
        decoded_sequences.append(decoded_sequence)
    return decoded_sequences


def decode_single_sequence(sequence):
    current_char = None
    decoded_chars = []
    for index in sequence:
        predicted_char = INDEX2LETTER[index.item()]
        if predicted_char != current_char and predicted_char != EPSILON:
            decoded_chars.append(predicted_char)
            current_char = predicted_char
    decoded_sequence = "".join(decoded_chars)
    return decoded_sequence if decoded_sequence != "thre" else "three"


def evaluate_model_performance(ctc_model, data_loader, device):
    ctc_model.eval()
    successful_predictions = 0
    exact_matches = 0
    total_samples = 0
    ctc_loss = CTCWithLLM(blank=LETTER2INDEX[EPSILON])

    with torch.no_grad():

        for mfccs, labels, input_lengths, target_lengths in data_loader:
            mfccs = mfccs.to(device)
            labels = labels.to(device)
            logits = ctc_model(mfccs).permute(1, 0, 2)
            predictions = decode_output(logits)

            for i, (predicted_logit, prediction, label) in enumerate(
                    zip(logits.permute(1, 0, 2), predictions, labels)):
                lowest_loss = torch.inf
                best_prediction = None
                for digit in data_loader.dataset.digit_names:
                    class_indices = torch.tensor([LETTER2INDEX[char] for char in digit],
                                                 dtype=torch.long).to(device)
                    class_length = torch.tensor([len(class_indices)], dtype=torch.long).to(device)
                    class_specific_loss = ctc_loss(predicted_logit.unsqueeze(1),
                                                   class_indices.unsqueeze(0),
                                                   input_lengths[i:i + 1], class_length)
                    if class_specific_loss < lowest_loss:
                        lowest_loss = class_specific_loss
                        best_prediction = digit

                decoded_label = "".join([INDEX2LETTER[index.item()] for index in label
                                         if index.item() != LETTER2INDEX[EPSILON]])
                successful_predictions += int(best_prediction == decoded_label)
                exact_matches += int(prediction.strip() == decoded_label.strip())
                print(f"Predicted: {prediction}, Actual: {decoded_label}")
                total_samples += 1

    return successful_predictions / total_samples, exact_matches / total_samples


class CTCModel(nn.Module):

    def __init__(self, hidden_dim, mfcc_dim, output_dim):
        super(CTCModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=mfcc_dim, out_channels=hidden_dim,
                               kernel_size=CONV_KERNEL_SIZE, padding=PADDING)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=output_dim,
                               kernel_size=CONV_KERNEL_SIZE, padding=PADDING)
        self.activation = nn.LeakyReLU()
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, inputs):  # shape: (batch_size, time_dim, mfcc_dim)
        inputs = inputs.transpose(1, 2)  # afterwards, shape: (batch_size, mfcc_dim, time_dim)
        inputs = self.conv1(inputs)  # afterwards, shape: (batch_size, hidden_dim, time_dim)
        inputs = self.activation(inputs)
        inputs = self.conv2(inputs)  # afterwards, shape: (batch_size, output_dim, time_dim)
        inputs = inputs.transpose(1, 2)  # afterwards, shape: (batch_size, time_dim, output_dim)
        return self.softmax(inputs)


def build_datasets():
    train_dataset = DigitDataset(TRAIN_SET_PATH)
    val_dataset = DigitDataset(VAL_SET_PATH)
    test_dataset = DigitDataset(TEST_SET_PATH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda x: pad_tensors_and_vectorize_labels(x, train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=lambda x: pad_tensors_and_vectorize_labels(x, val_dataset))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=lambda x: pad_tensors_and_vectorize_labels(x, test_dataset))

    # final_max_steps = (f"\nMIGHT SAVE TIME: max_time_steps for train, val, test: "
    #                    f"{train_dataset.max_time_steps}, {val_dataset.max_time_steps}, "
    #                    f"{test_dataset.max_time_steps}\n")
    # print(final_max_steps)
    # with open("final_max_steps.txt", "w") as f:
    #     f.write(final_max_steps)

    return train_loader, val_loader, test_loader


class CTCWithLLM(nn.CTCLoss):

    def __init__(self, blank):
        super().__init__(blank)
        # self.pipeline = pipeline("text-generation", model=LLM_NAME)
        self.llm = ChatOpenAI(model_name=LLM_NAME) if "OPENAI_API_KEY" in os.environ else None
        self.num_failed_attempts = 0
        self.llm_performance_logs_path = f"/cs/snapless/gabis/nive/speech/Speech-Processing-Project/{LOGS_DIR}/llm_performance_logs_{time.strftime('%H_%M_%S')}.txt"
        with open(self.llm_performance_logs_path, "w") as f:
            f.write("Decoded CTC model's output,\tLLM Predicted Label,\tOriginal True Label,\tAre they the same\n")

    def get_targets_by_llm(self, logits: torch.Tensor, target: torch.Tensor):
        decoded_output = decode_output(logits.unsqueeze(1))[0]
        # Create the prompt for the LLM
        input_text = LLM_PROMPT.format(decoded_output=decoded_output)
        # Use the pipeline to generate a response from the LLM
        # response = self.pipeline(input_text, max_length=50, num_return_sequences=1)[0]['generated_text']
        try:
            response = self.llm(input_text).content.lower().strip()
        except Exception:
            time.sleep(5)
            try:
                response = self.llm(input_text).content.lower().strip()
            except Exception:
                response = "Langchain Error"
        label = None
        for digit_name in DIGITS:
            if digit_name in response or response in digit_name:
                label = digit_name
                break
        if label is None:
            label = random.choice(DIGITS)
            self.num_failed_attempts += 1
        decoded_true_label = decode_single_sequence(target * (-1))
        with open(self.llm_performance_logs_path, "a") as f:
            f.write(f"{decoded_output},\t{response},\t{decoded_true_label},\t{response == decoded_true_label}\n")
        return labels_to_padded_targets([label])

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        for i, target in enumerate(targets):
            if (target < 0).any():
                if self.llm is None:
                    targets[i] *= (-1)
                else:  # You should define 'OPENAI_API_KEY' in os.environ to use the LLM
                    targets[i] = self.get_targets_by_llm(log_probs[:, i, :], target)
        return super().forward(log_probs, targets, input_lengths, target_lengths)

    def log_epoch_number(self, epoch_number):
        with open(self.llm_performance_logs_path, "a") as f:
            f.write(f"Epoch: {epoch_number}\n")


def main():

    # all kinds of random seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)  # when using a GPU, set the seed for all GPU operations
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)  # when using multiple GPUs


    # create and load datasets
    print("Started running at", time.strftime("%H:%M:%S"), flush=True)
    train_loader, val_loader, test_loader = build_datasets()
    print("Datasets loaded successfully at", time.strftime("%H:%M:%S"), flush=True)

    # init model and its auxiliary tools
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CTCModel(hidden_dim=HIDDEN_DIM, mfcc_dim=N_MFCC, output_dim=len(LETTER2INDEX)).to(device)
    criterion = CTCWithLLM(blank=LETTER2INDEX[EPSILON])
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print("Model initiated.", flush=True)

    logs = f""" {time.strftime("%H:%M:%S")}
    NUM_EPOCHS = {NUM_EPOCHS}
    NUM_CLEAN_EPOCHS = {NUM_CLEAN_EPOCHS}
    DONT_USE_LABELS_RATIO = {DONT_USE_LABELS_RATIO}
    LLM_NAME = "{LLM_NAME}"
    LLM_PROMPT = "{LLM_PROMPT}"
    \n
    """
    # training
    print("Starts training at", time.strftime("%H:%M:%S"), flush=True)
    for epoch in tqdm(range(NUM_EPOCHS)):
        print(f"Starts epoch #{epoch + 1} at", time.strftime("%H:%M:%S"), flush=True)
        criterion.log_epoch_number(epoch + 1)
        model.train()
        aggregated_loss = 0
        for mfccs, labels, input_lengths, target_lengths in train_loader:
            labels[::DONT_USE_LABELS_RATIO] *= -1
            if epoch < NUM_CLEAN_EPOCHS or REFERENCE_RUN_WITH_HALF_DATA:
                samples_to_keep = (labels > 0)[:, 0]
                mfccs = mfccs[samples_to_keep]
                labels = labels[samples_to_keep]
                input_lengths = input_lengths[samples_to_keep]
                target_lengths = target_lengths[samples_to_keep]
            mfccs = mfccs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(mfccs).transpose(0, 1)
            loss = criterion(outputs, labels, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            aggregated_loss += loss.item()

        # saving checkpoint in case of a crash in the code:
        if epoch % CHECKPOINT_INTERVAL == CHECKPOINT_INTERVAL - 1 or epoch == NUM_EPOCHS - 1:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, f"{LOGS_DIR}/checkpoint_epoch_{epoch + 1}.pth.tar")

        # validating
        with torch.inference_mode():
            accuracy, exact_match_accuracy = evaluate_model_performance(model, val_loader, device)
            status = (f"Finished Epoch {epoch + 1} out of {NUM_EPOCHS}\n"
                      f"Mean CTC loss: {aggregated_loss / len(train_loader)}\n"
                      f"Accuracy over validation set: {accuracy}\n"
                      f"Exact-match accuracy over validation set: {exact_match_accuracy}\n")
            if epoch >= NUM_CLEAN_EPOCHS:
                status += f"Rate of failed LLM attempts: (random choice instead) " \
                          f"{criterion.num_failed_attempts / len(train_loader.dataset)}\n"
                criterion.num_failed_attempts = 0
            print(status, flush=True)
            logs += status

    # testing
    with torch.inference_mode():
        final_accuracy, final_exact_match_accuracy = evaluate_model_performance(model, test_loader, device)
        final_status = (f"Finished training at {time.strftime('%H:%M:%S')}\n"
                        f"Final accuracy: {final_accuracy}\n"
                        f"Final exact-match accuracy: {final_exact_match_accuracy}\n"
                        f"LLM performance logs path: {criterion.llm_performance_logs_path}")
        print(final_status)
        logs += final_status
        logs_output_path = f"/cs/snapless/gabis/nive/speech/Speech-Processing-Project/{LOGS_DIR}/output_logs_{time.strftime('%H_%M_%S')}.txt"
        with open(logs_output_path, "w") as f:
            f.write(logs)
        print(f"Full logs were saved to {logs_output_path}", flush=True)


if __name__ == '__main__':
    main()
