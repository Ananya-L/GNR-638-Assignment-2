from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch

def plot_pca_features(model, loader, device, model_name):

    model.eval()

    features = []
    labels = []

    with torch.no_grad():

        for images, y in loader:

            images = images.to(device)

            feats = model.forward_features(images)

            # handle models that output feature maps
            if len(feats.shape) == 4:
                feats = feats.mean(dim=[2,3])  # global average pooling

            features.append(feats.cpu())
            labels.append(y)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    # limit samples for faster PCA
    if len(features) > 900:
        features = features[:900]
        labels = labels[:900]

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(features)

    plt.figure(figsize=(8,6))

    plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap="tab20", s=10)

    plt.title(f"PCA Feature Embedding - {model_name}")

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.savefig(f"pca_features_{model_name}.png")

    plt.show()