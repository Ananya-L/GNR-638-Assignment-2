import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import torch
import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from models import resnet50, densenet121, efficientnet_b0


# -----------------------
# Save to Google Drive
# -----------------------

SAVE_DIR = "/content/drive/MyDrive/GNR638_results"

os.makedirs(SAVE_DIR, exist_ok=True)
clean_accuracy = {}

# -----------------------
# Corruption Classes
# -----------------------

class GaussianNoise:

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):

        noise = torch.randn_like(img) * self.sigma
        img = img + noise

        return torch.clamp(img,0,1)


class BrightnessShift:

    def __init__(self,factor):
        self.factor = factor

    def __call__(self,img):

        return F.adjust_brightness(img,self.factor)


class MotionBlur:

    def __init__(self,kernel_size):
        self.kernel_size = kernel_size

    def __call__(self,img):

        img = np.array(img)

        kernel = np.zeros((self.kernel_size,self.kernel_size))
        kernel[int((self.kernel_size-1)/2),:] = np.ones(self.kernel_size)
        kernel = kernel/self.kernel_size

        blurred = cv2.filter2D(img,-1,kernel)

        return blurred


# -----------------------
# Evaluation
# -----------------------

def evaluate(model,loader,device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images,labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct/total


# -----------------------
# Experiment
# -----------------------

def run_corruption_test():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = "dataset/val"

    base_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    clean_dataset = datasets.ImageFolder(dataset_path,transform=base_transform)
    clean_loader = DataLoader(clean_dataset,batch_size=32,shuffle=False)

    models = {
        "resnet50": resnet50(),
        "densenet121": densenet121(),
        "efficientnet_b0": efficientnet_b0()
    }

    results = []
    time_log = {}

    for name,model in models.items():

        start_time = time.time()

        print("\nRunning:",name)

        model = model.to(device)

        clean_acc = evaluate(model,clean_loader,device)
        clean_accuracy[name] = clean_acc
        # ----------------
        # Gaussian Noise
        # ----------------

        for sigma in [0.05,0.1,0.2]:

            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                GaussianNoise(sigma)
            ])

            dataset = datasets.ImageFolder(dataset_path,transform=transform)
            loader = DataLoader(dataset,batch_size=32)

            acc = evaluate(model,loader,device)

            results.append({
                "model":name,
                "corruption":"gaussian",
                "level":sigma,
                "accuracy":acc,
                "corruption_error":1-acc,
                "relative_robustness":acc/clean_acc
            })

        # ----------------
        # Motion Blur
        # ----------------

        for k in [5,9,13]:

            transform = transforms.Compose([
                transforms.Resize((224,224)),
                MotionBlur(k),
                transforms.ToTensor()
            ])

            dataset = datasets.ImageFolder(dataset_path,transform=transform)
            loader = DataLoader(dataset,batch_size=32)

            acc = evaluate(model,loader,device)

            results.append({
                "model":name,
                "corruption":"motion_blur",
                "level":k,
                "accuracy":acc,
                "corruption_error":1-acc,
                "relative_robustness":acc/clean_acc
            })

        # ----------------
        # Brightness
        # ----------------

        for b in [0.5,1.5,2.0]:

            transform = transforms.Compose([
                transforms.Resize((224,224)),
                BrightnessShift(b),
                transforms.ToTensor()
            ])

            dataset = datasets.ImageFolder(dataset_path,transform=transform)
            loader = DataLoader(dataset,batch_size=32)

            acc = evaluate(model,loader,device)

            results.append({
                "model":name,
                "corruption":"brightness",
                "level":b,
                "accuracy":acc,
                "corruption_error":1-acc,
                "relative_robustness":acc/clean_acc
            })


        end_time = time.time()

        time_log[name] = end_time - start_time


    df = pd.DataFrame(results)

    df.to_csv(f"{SAVE_DIR}/corruption_results.csv",index=False)
    clean_df = pd.DataFrame.from_dict(clean_accuracy, orient="index", columns=["clean_accuracy"])
    clean_df.to_csv(f"{SAVE_DIR}/clean_accuracy.csv")

    summary = df.groupby(["model","corruption"]).agg({
    "accuracy":"mean",
    "corruption_error":"mean",
    "relative_robustness":"mean"
}).reset_index()

    summary.to_csv(f"{SAVE_DIR}/corruption_summary.csv",index=False)

    # -----------------------
    # Save runtime log
    # -----------------------

    with open(f"{SAVE_DIR}/runtime_log.txt","w") as f:

        for k,v in time_log.items():

            f.write(f"{k} runtime: {v/60:.2f} minutes\n")

    # -----------------------
    # Separate plots per model
    # -----------------------

    for model in df["model"].unique():

        model_df = df[df["model"]==model]

        plt.figure()

        sns.lineplot(
            data=model_df,
            x="level",
            y="accuracy",
            hue="corruption",
            marker="o"
        )

        plt.title(f"Robustness Analysis - {model}")

        plt.savefig(f"{SAVE_DIR}/{model}_robustness.png")

        plt.close()

    for corruption in df["corruption"].unique():

        plt.figure()

        subset = df[df["corruption"]==corruption]

        sns.lineplot(
            data=subset,
            x="level",
            y="accuracy",
            hue="model",
            marker="o"
        )

        plt.title(f"Model Comparison - {corruption}")

        plt.savefig(f"{SAVE_DIR}/{corruption}_comparison.png")

        plt.close()

    with open(f"{SAVE_DIR}/experiment_log.txt","w") as f:

        f.write("Corruption Robustness Experiment\n\n")

        f.write("Models evaluated:\n")
        for m in models.keys():
            f.write(m+"\n")

        f.write("\nCorruption Levels:\n")
        f.write("Gaussian: 0.05, 0.1, 0.2\n")
        f.write("Motion Blur: 5, 9, 13\n")
        f.write("Brightness: 0.5, 1.5, 2.0\n")

        f.write("\nRuntime per model:\n")

        for k,v in time_log.items():
            f.write(f"{k}: {v/60:.2f} minutes\n")

        print("All results saved to:",SAVE_DIR)


# -----------------------
# Run
# -----------------------

if __name__ == "__main__":

    run_corruption_test()
