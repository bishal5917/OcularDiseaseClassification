
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import dataloader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.current_device())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device = {device}")

# setting seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)

#setting the hyperparameters
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
PATCH_SIZE = 4
IMAGE_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 4
EMBED_DIM = 256
NUM_HEADS = 8
DEPTH = 6
MLP_DIM = 512
DROPOUT_RATE = 0.5


