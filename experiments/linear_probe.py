import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.metrics import plot_confusion_matrix
from utils.plots import plot_pca_features
from models.model_loader import freeze_backbone, load_model
from utils.dataset import get_dataloaders
from training.train import train_one_epoch
from training.evaluate import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader, num_classes = get_dataloaders("dataset")

models_to_run = [
    "resnet50",
    "densenet121",
    "efficientnet_b0"
]

epochs = 30

for model_name in models_to_run:
    torch.cuda.empty_cache()
    print(f"\n========== Training {model_name} ==========\n")

    model = load_model(model_name, num_classes)

    freeze_backbone(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_accs = []
    val_accs = []

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_acc = evaluate(model, val_loader, device)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"{model_name} Epoch {epoch + 1}: "
            f"Train Acc {train_acc:.3f}, "
            f"Val Acc {val_acc:.3f}"
        )
    torch.save(
    model.state_dict(),
    f"/content/drive/MyDrive/{model_name}_linear_probe.pth"
    )
    # Plot accuracy curve
    plt.figure()

    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Linear Probe - {model_name}")

    plt.legend()

    plt.savefig(f"linear_probe_{model_name}_accuracy.png")
    plt.savefig(f"/content/drive/MyDrive/linear_probe_{model_name}_accuracy.png")

    plt.show()

    class_names = train_loader.dataset.classes

    plot_confusion_matrix(model, val_loader, device, class_names, model_name)

    plot_pca_features(model, val_loader, device, model_name)