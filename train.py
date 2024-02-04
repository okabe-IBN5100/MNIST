import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Classifier

from tqdm import tqdm

from data import trainloader
from config import *

import matplotlib.pyplot as plt

if __name__ == "__main__":
    loss = []

    model = Classifier().to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"DEVICE IS SET TO {DEVICE}",end="\n")

    model.train(True)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}...")
        for batch_id, (inputs, labels) in enumerate(tqdm(trainloader)):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            labels = F.one_hot(labels, num_classes=10).float()

            optimizer.zero_grad()
            out = model(inputs)
            L = criterion(out, labels)

            L.backward()
            optimizer.step()

            loss.append(L.item())

    torch.save(model, "model.pt")
    plt.plot(loss)
    plt.show()

