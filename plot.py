
import os
import matplotlib.pyplot as plt

def plot_predictions(images, true_labels, predicted_labels, class_names, show_plot=False, save_plot=True):
    grid_size = (2, 2)
    rows, cols = grid_size
    num_samples = len(images)

    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 7*rows))
    axes = axes.flatten()  # flatten to 1D for easy indexing

    for i in range(len(axes)):
        ax = axes[i]
        if i < num_samples:
            # Unnormalize image
            img = unnormalize(images[i])
            # img = images[i]
            ax.imshow(img)
            true_cls = class_names[true_labels[i]]
            pred_cls = class_names[predicted_labels[i]]
            color = "green" if true_cls == pred_cls else "red"
            ax.set_title(f"TRUE: {true_cls}   PREDICTED: {pred_cls}", color=color, fontsize=24)
        ax.axis("off")  # hide unused axes or axis lines

    plt.suptitle(f"Classification Results", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if save_plot:
        if not os.path.exists('results'):
            os.makedirs('results')
        plt.savefig(f'results/prediction.png')

    if show_plot:
        plt.show()

def unnormalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    img_tensor: shape (C,H,W)
    returns img: (H,W,C) in range 0-1
    """
    img = img_tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]  # reverse normalization
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
    return img


def plot_loss(train_losses, validation_losses, model_name):
    plt.figure(figsize=(15, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(validation_losses, label="Validation Loss")

    plt.gca().set_xlabel("Epoch")
    plt.gca().set_ylabel("Loss")
    plt.gca().set_title(f"{model_name} Loss Plot")

    plt.legend()

    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(f'plots/{model_name}_loss_plot.png')
    # plt.show()

def plot_accuracy(train_accuracies, validation_accuracies, model_name):
    plt.figure(figsize=(15, 5))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(validation_accuracies, label="Validation Accuracy")

    plt.gca().set_xlabel("Epoch")
    plt.gca().set_ylabel("Accuracy")
    plt.gca().set_title(f"{model_name} Accuracy Plot")

    plt.legend()

    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(f'plots/{model_name}_accuracy_plot.png')
    # plt.show()




