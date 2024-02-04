import torch

EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
