import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# print(f"Pytorch version: {torch.__version__}")
# print(f"Training on: {'GPU' if torch.cuda.is_available() else 'CPU'}")

with  open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_number = {}
number_to_char = {}

for i, char in enumerate(chars):
    char_to_number[char] = i
    number_to_char[i] = char


def encode(words):
    tokenised_words = list(words)
    word_vector = []
    for char in tokenised_words:
        word_vector.append(char_to_number[char])
    return word_vector

def decode(numbers):
    numbers_vector = []
    for num in numbers:
        numbers_vector.append(number_to_char[num])

    text = ''.join(numbers_vector)
    return text

data = encode(text)

n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

batch_size = 32
block_size = 64

def get_batch(split):
    data = train_data if split == 'train' else val_data
    highest_low = len(data) - block_size
    input_tensor = []
    target_tensor = []
    ix = torch.randint(0, highest_low, (batch_size,))

    for i in range(batch_size):
        pos = ix[i]
        input_chunk = data[pos:pos+block_size]
        target_chunk = data[pos+1: pos+block_size+1]

        tensored_input_chunk = torch.tensor(input_chunk)
        tensored_target_chunk = torch.tensor(target_chunk)

        input_tensor.append(tensored_input_chunk)
        target_tensor.append(tensored_target_chunk)

    input_stack = torch.stack(input_tensor)
    target_stack = torch.stack(target_tensor)
    return input_stack, target_stack
