import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from models.model_loader import load_model
from utils.dataset import get_dataloaders
from utils.feature_extractor import extract_features
from training.train_linear_probe import train_probe



save_dir = "/content/drive/MyDrive/GNR638_results"
os.makedirs(save_dir, exist_ok=True)

def get_balanced_subset(features, labels, samples_per_class=30):

    selected_feats = []
    selected_labels = []

    for c in torch.unique(labels):

        idx = (labels == c).nonzero(as_tuple=True)[0][:samples_per_class]

        selected_feats.append(features[idx])
        selected_labels.append(labels[idx])

    feats = torch.cat(selected_feats)
    labs = torch.cat(selected_labels)

    return feats, labs

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader, num_classes = get_dataloaders("dataset")


# ---------------------------------
# Create fixed PCA subset (30x30)
# ---------------------------------

subset_indices = []

label_list = []

with torch.no_grad():
    for _, y in val_loader:
        label_list.append(y)

labels = torch.cat(label_list)

for c in torch.unique(labels):

    idx = (labels == c).nonzero(as_tuple=True)[0][:30]

    subset_indices.append(idx)

subset_indices = torch.cat(subset_indices)



models = {
    "resnet50": ["layer1","layer3","layer4"],
    "densenet121": ["features.denseblock1",
                    "features.denseblock3",
                    "features.denseblock4"],
    "efficientnet_b0": ["blocks.1","blocks.4","blocks.6"]
}

results = []
pca_storage = {}
for model_name,layers in models.items():

    model = load_model(model_name,num_classes)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    for depth,layer in enumerate(layers):

        train_feats,train_labels = extract_features(
            model,layer,train_loader,device
        )

        feature_norm = train_feats.norm(dim=1).mean().item()



        val_feats,val_labels = extract_features(
            model,layer,val_loader,device
        )
        cluster_variances = []

        for c in torch.unique(val_labels):

            idx = (val_labels == c).nonzero(as_tuple=True)[0]

            class_feats = val_feats[idx]

            var = class_feats.var(dim=0).mean().item()

            cluster_variances.append(var)

        cluster_compactness = np.mean(cluster_variances)
        # PCA visualization for this layer
        subset_feats = val_feats[subset_indices]
        subset_labels = val_labels[subset_indices]
        subset_feats = subset_feats.cpu()
        pca = PCA(n_components=2)

        reduced = pca.fit_transform(subset_feats.numpy())

        # store PCA for combined plot
        pca_storage[(model_name, depth)] = (reduced, subset_labels)

        plt.figure(figsize=(7,4))

        plt.scatter(
            reduced[:,0],
            reduced[:,1],
            c=subset_labels.cpu().numpy(),
            cmap="tab20",
            s=10
        )

        plt.title(f"PCA - {model_name} - {layer}")

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)

        plt.savefig(f"pca_{model_name}_{layer}.png", dpi=300)
        plt.savefig(os.path.join(save_dir, f"pca_{model_name}_{layer}.png"), dpi=300)

        plt.close()

        clf = train_probe(train_feats,train_labels,num_classes)

        preds = clf(val_feats.to(device)).argmax(1).cpu()

        acc = (preds==val_labels).float().mean().item()
        
        feature_std = train_feats.std().item()
        results.append({
            "model":model_name,
            "layer":layer,
            "depth":depth,
            "accuracy":acc,
            "feature_norm":feature_norm,
            "cluster_compactness":cluster_compactness,
            "params":total_params,
            "feature_std":feature_std
        })

for model_name in models.keys():

    fig, axes = plt.subplots(1,3,figsize=(12,4))

    depth_names = ["Early","Middle","Final"]
    

    for d in range(3):

        reduced, labels = pca_storage[(model_name,d)]

        axes[d].scatter(
            reduced[:,0],
            reduced[:,1],
            c=labels.cpu().numpy(),
            cmap="tab20",
            s=8
        )

        axes[d].set_title(depth_names[d])
        axes[d].set_xlabel("PC1")
        axes[d].set_ylabel("PC2")

    fig.suptitle(f"PCA Across Network Depth - {model_name}")

    plt.tight_layout()

    plt.savefig(f"pca_depth_{model_name}.png",dpi=300)
    plt.savefig(os.path.join(save_dir, f"pca_depth_{model_name}.png"), dpi=300)

    plt.close()
df = pd.DataFrame(results)
depth_labels = {0: "Early", 1: "Middle", 2: "Final"}
df["depth_name"] = df["depth"].map(depth_labels)
df["depth_name"] = pd.Categorical(
    df["depth_name"],
    categories=["Early","Middle","Final"],
    ordered=True
)
df.to_csv(os.path.join(save_dir, "layer_probe_results.csv"),index=False)





plt.figure(figsize=(7,4))

sns.lineplot(
    data=df,
    x="depth_name",
    y="accuracy",
    hue="model",
    marker="o",
    linewidth = 2
)

plt.xlabel("Network Depth")
plt.ylabel("Validation Accuracy")
plt.title("Layer-wise Representation Quality")

plt.grid(True)

plt.savefig("layer_probe_accuracy.png", dpi=300)
plt.savefig(os.path.join(save_dir, "layer_probe_accuracy.png"), dpi=300)

plt.show()

# -----------------------
# STEP 5: Feature Norm Table
# -----------------------

norm_table = df.pivot_table(
    index="model",
    columns="depth_name",
    values="feature_norm"
)

print("\nFeature Norm Statistics")
print(norm_table)

norm_table.to_csv(os.path.join(save_dir, "feature_norm_statistics.csv"))

compactness_table = df.pivot_table(
    index="model",
    columns="depth_name",
    values="cluster_compactness"
)

print("\nCluster Compactness")
print(compactness_table)

compactness_table.to_csv(os.path.join(save_dir, "cluster_compactness.csv"))
