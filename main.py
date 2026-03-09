import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models.model_loader import freeze_backbone, load_model
from utils.dataset import get_dataloaders
from training.train import train_one_epoch
from training.evaluate import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"


train_loader, val_loader, num_classes = get_dataloaders("dataset")

model = load_model("resnet50", num_classes)


freeze_backbone(model)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)


epochs = 30

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
        f"Epoch {epoch}: "
        f"Train Acc {train_acc:.3f}, "
        f"Val Acc {val_acc:.3f}"
    )



plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Linear Probe Transfer Learning")

plt.legend()
plt.show()
plt.savefig("linear_probe_resnet50_accuracy.png")