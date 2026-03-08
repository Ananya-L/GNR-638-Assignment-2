import torch
import torch.nn as nn
import torch.optim as optim

from models.model_loader import freeze_backbone, load_model
from utils.dataset import get_dataloaders
from training.train import train_one_epoch
from training.evaluate import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"


train_loader, val_loader, num_classes = get_dataloaders("dataset")

model = load_model("resnet50", num_classes)


freeze_backbone(model)

model = model.to(device)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)


epochs = 5

for epoch in range(epochs):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device
    )

    val_acc = evaluate(model, val_loader, device)

    print(
        f"Epoch {epoch}: "
        f"Train Acc {train_acc:.3f}, "
        f"Val Acc {val_acc:.3f}"
    )