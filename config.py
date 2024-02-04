import torch

EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.001
BATCH_SIZE = 64
