from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch

def plot_confusion_matrix(model, loader, device, class_names, model_name):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            preds = outputs.argmax(1).cpu()

            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10,8))

    sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"confusion_matrix_{model_name}.png")

    plt.show()