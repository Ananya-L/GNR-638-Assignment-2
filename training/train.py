import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, preds = outputs.max(1)

        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total

    return total_loss / len(loader), accuracy