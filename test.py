import sys

import torch
import random
from model import VisionTransformer
from data import test_loader
from plot import plot_predictions

def test():
    model_path = f'models/ViT_model.pth'
    CLASS_NAMES = ['amd', 'cataract', 'diabetes', 'normal']

    PATCH_SIZE = 16
    IMAGE_SIZE = 224
    CHANNELS = 3
    NUM_CLASSES = 4
    EMBED_DIM = 128
    NUM_HEADS = 8
    DEPTH = 6
    MLP_DIM = 256
    DROPOUT_RATE = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"---------- Testing Model ----------")

    # Instantiate model
    model = VisionTransformer(IMAGE_SIZE, PATCH_SIZE, CHANNELS, NUM_CLASSES,
                              EMBED_DIM, DEPTH, NUM_HEADS, MLP_DIM, DROPOUT_RATE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get dataset from test_loader
    test_dataset = test_loader.dataset
    total_samples = len(test_dataset)

    random_indices = random.sample(range(total_samples), 4)
    print(f"random_indices: {random_indices}")

    # Collect images, labels, predictions
    all_images = []
    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for idx in random_indices:
            img, label = test_dataset[idx]

            # print(img.shape)
            input_tensor = img.unsqueeze(dim=0).to(device)
            # print(input_tensor.shape)

            label = torch.tensor([label]).to(device)

            output = model(input_tensor)
            _, predicted = torch.max(output.data, 1)

            all_images.append(img.cpu())
            all_labels.append(label.item())
            all_predictions.append(predicted.item())

    print(f"all_labels: {all_labels}")
    print(f"all_predictions: {all_predictions}")

    match_count = 0

    for i in range(len(all_labels)):
        true_cls = CLASS_NAMES[all_labels[i]]
        predicted_cls = CLASS_NAMES[all_predictions[i]]
        if true_cls == predicted_cls:
            match_count += 1
        print(f"Sample {i + 1}: True = {true_cls}, Predicted = {predicted_cls}")

    if match_count == 4:
        print("All samples match!")

    # sys.exit()
    # Plot predictions using matplotlib
    plot_predictions(
        images=all_images,
        true_labels=all_labels,
        predicted_labels=all_predictions,
        class_names=CLASS_NAMES,
        save_plot=False,
        show_plot=True,
    )

test()
