import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt


from models.model_loader import load_model
from utils.dataset import get_dataloaders, get_subset_loader
from training.train import train_one_epoch
from training.evaluate import evaluate




save_dir = "/content/drive/MyDrive/GNR638_results"
os.makedirs(save_dir, exist_ok=True)


log_file = open(f"{save_dir}/training_log.txt","w")
sys.stdout = log_file

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader, num_classes = get_dataloaders("dataset")

dataset = train_loader.dataset

fractions = [0.05, 0.2, 1.0]
fractions_percent = [5, 20, 100]

models_to_run = [
    "resnet50",
    "densenet121",
    "efficientnet_b0"
]

all_results = {}
gap_results = {}
relative_drop = {}

for model_name in models_to_run:

    print(f"\n================ {model_name} =================\n")

    results = {}
    gaps = {}

    for frac in fractions:

        print(f"\n===== Training with {int(frac*100)}% data =====")

        subset = get_subset_loader(dataset, frac)

        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=32,
            shuffle=True
        )

        model = load_model(model_name, num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        epochs = 20

        for epoch in range(epochs):

            train_loss, train_acc = train_one_epoch(
                model,
                loader,
                optimizer,
                criterion,
                device
            )

        val_acc = evaluate(model, val_loader, device)

        gap = train_acc - val_acc

        print(f"Train Accuracy: {train_acc:.3f}")
        print(f"Validation Accuracy: {val_acc:.3f}")
        print(f"Training–Validation Gap: {gap:.3f}")

        results[frac] = (train_acc, val_acc)
        gaps[frac] = gap

    all_results[model_name] = results
    gap_results[model_name] = gaps

    print("\nFinal Results:")

    for k, v in results.items():
        train_acc, val_acc = v
        print(f"{int(k*100)}% data → Train: {train_acc:.3f}, Val: {val_acc:.3f}")

    acc100 = results[1.0][1]
    acc5 = results[0.05][1]

    delta = (acc100 - acc5) / acc100
    relative_drop[model_name] = delta

    print("Relative Performance Drop:", delta)
    model_path = f"{save_dir}/{model_name}_fewshot_model.pth"
    torch.save(model.state_dict(), model_path)

    print("Model saved to:", model_path)

    accuracies = [
        results[0.05][1],
        results[0.2][1],
        results[1.0][1]
    ]

    plt.figure()

    plt.plot(fractions_percent, accuracies, marker='o')

    plt.xlabel("Training Data (%)")
    plt.ylabel("Validation Accuracy")
    plt.title(f"Few-Shot Learning Performance - {model_name}")
    plt.grid(True)
    plt.savefig(f"{save_dir}/few_shot_{model_name}.png")
    plt.savefig(f"few_shot_{model_name}.png")
    plt.show()


# -------------------------
# Combined Accuracy Plot
# -------------------------

plt.figure()

for model_name in models_to_run:

    res = all_results[model_name]

    accuracies = [
        res[0.05][1],
        res[0.2][1],
        res[1.0][1]
    ]

    plt.plot(
        fractions_percent,
        accuracies,
        marker='o',
        label=model_name
    )

plt.xlabel("Training Data (%)")
plt.ylabel("Validation Accuracy")
plt.title("Few-Shot Learning Comparison")

plt.legend()
plt.grid(True)
plt.savefig(f"{save_dir}/few_shot_combined.png")
plt.savefig("few_shot_combined.png")
plt.show()


# -------------------------
# Combined Gap Plot
# -------------------------

plt.figure()

for model_name in models_to_run:

    gaps = [
        gap_results[model_name][0.05],
        gap_results[model_name][0.2],
        gap_results[model_name][1.0]
    ]

    plt.plot(
        fractions_percent,
        gaps,
        marker='o',
        label=model_name
    )

plt.xlabel("Training Data (%)")
plt.ylabel("Training–Validation Gap")
plt.title("Overfitting Analysis")

plt.legend()
plt.grid(True)
plt.savefig(f"{save_dir}/few_shot_gap_analysis.png")
plt.savefig("few_shot_gap_analysis.png")
plt.show()


# -------------------------
# Print Relative Drops
# -------------------------

print("\nRelative Performance Drop for Each Model")

for model in models_to_run:
    print(model, "→", round(relative_drop[model],3))

print("\nSummary Table")

for model in models_to_run:
    for frac in fractions:
        train_acc, val_acc = all_results[model][frac]
        gap = train_acc - val_acc
        print(model, frac, train_acc, val_acc, gap)

import pandas as pd

rows = []

for model in models_to_run:
    for frac in fractions:
        train_acc, val_acc = all_results[model][frac]
        gap = train_acc - val_acc

        rows.append({
            "model": model,
            "data_fraction": frac,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "gap": gap
        })

df = pd.DataFrame(rows)

df["relative_drop"] = df["model"].map(relative_drop)

df.to_csv(f"{save_dir}/few_shot_results.csv", index=False)

print("Results saved to CSV")

sys.stdout = sys.__stdout__
log_file.close()

print("Training finished. Results saved to Drive.")