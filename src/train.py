import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import IntrusionNet


def train():
    device = torch.device("cpu")
    torch.set_num_threads(4)

    train_loader, test_loader, input_dim, num_classes = get_dataloaders(
        "../data/nslkdd/KDDTrain+.txt",
        "../data/nslkdd/KDDTest+.txt",
        batch_size=64
    )

    model = IntrusionNet(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 25

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "../results/model.pth")
    print("Model saved to results/model.pth")


if __name__ == "__main__":
    train()