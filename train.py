import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

from ViT_model import VisionTransformer
from tqdm.auto import tqdm
from data import train_loader, val_loader
from plot import plot_loss, plot_accuracy

from ResNet_model import ResNet50, ResNet101, ResNet152


def train_model(model, device, loader, optimizer, criterion):
    # setting into training mode
    model.train()

    total_loss, correct = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # forward pass
        output = model(x)
        loss = criterion(output, y)

        # backward prop
        loss.backward()

        # gradient descent
        optimizer.step()

        # accumulate the loss over the batch
        total_loss += loss.item() * x.size(0)
        correct += (output.argmax(1) == y).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate_model(model, device, loader, criterion):
    # setting into evaluation mode
    model.eval()
    total_loss, correct = 0, 0

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)

            total_loss += loss.item() * x.size(0)
            correct += (output.argmax(1) == y).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def train(model_name = "ViT"):
    # setting the hyperparameters
    EPOCHS = 20
    if model_name == "ViT":
        LEARNING_RATE = 1e-3
    else:
        LEARNING_RATE = 1e-4
    PATCH_SIZE = 16
    IMAGE_SIZE = 224
    CHANNELS = 3
    NUM_CLASSES = 4
    EMBED_DIM = 128
    NUM_HEADS = 8
    DEPTH = 6
    MLP_DIM = 256
    DROPOUT_RATE = 0.5

    print(f"Is CuDA available: {torch.cuda.is_available()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device = {device}")

    # setting seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    print(f"---------- Training Model {model_name} ----------")

    # Instantiate the model
    if model_name == "ViT":
        model = VisionTransformer(IMAGE_SIZE, PATCH_SIZE, CHANNELS, NUM_CLASSES,
                              EMBED_DIM, DEPTH, NUM_HEADS, MLP_DIM, DROPOUT_RATE)

    elif model_name == "ResNet50":
        model = ResNet50(img_channels=CHANNELS, num_classes=NUM_CLASSES)

    elif model_name == "ResNet101":
        model = ResNet101(img_channels=CHANNELS, num_classes=NUM_CLASSES)

    elif model_name == "ResNet152":
        model = ResNet152(img_channels=CHANNELS, num_classes=NUM_CLASSES)

    else:
        raise ValueError("Invalid model name")

    print(model)

    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(params=model.parameters(), lr = LEARNING_RATE)
    # print(optimizer)

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    best_val_accuracy = 0

    for epoch in tqdm(range(EPOCHS)):
        train_loss, train_accuracy = train_model(model, device, train_loader, optimizer, criterion)
        val_loss, val_accuracy = evaluate_model(model, device, val_loader, criterion)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"""
        Model: {model_name}
        Epoch {epoch + 1}/{EPOCHS}
        Train Accuracy: {train_accuracy:.4f}
        Train Loss: {train_loss:.4f}
        Validation Accuracy: {val_accuracy:.4f}
        Validation Loss: {val_loss:.4f}

        ----------------------------------------
        """)
        # getting the model with the best dice score
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), f"models/{model_name}_model.pth")

    # plotting the curves
    plot_accuracy(train_accuracies, val_accuracies, model_name)
    plot_loss(train_losses, val_losses, model_name)


train(model_name="ViT")
train(model_name="ResNet50")
train(model_name="ResNet101")
train(model_name="ResNet152")