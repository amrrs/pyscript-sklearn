import matplotlib.patches as mpatches
from matplotlib import figure
from matplotlib import pyplot as plt


def summarize_training(epochs: int, loss_train: list, accuracy_train: list, model_save_path: str) -> None:
    print(f"Model trained for {epochs} epochs.")
    print(f"Final training Loss: {loss_train[-1]}")
    print(f"Final training Accuracy: {accuracy_train[-1]}")
    print(f"Trained model saved under path {model_save_path}")


def visualize_training(loss_train: list, accuracy_train: list) -> figure.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(loss_train, color='blue')
    ax.plot(accuracy_train, color='orange')
    ax.set_title('Training over Time', size=16)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Value')
    orange_patch = mpatches.Patch(color='orange', label='Accuracy')
    blue_patch = mpatches.Patch(color='blue', label='Log Loss')
    ax.legend(handles=[orange_patch, blue_patch])
    return fig