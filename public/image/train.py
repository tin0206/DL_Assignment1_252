import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from dataset import CIFAR10Dataset, get_transforms
from models import get_model
from utils import evaluate
import time, pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10

os.makedirs("models", exist_ok=True)

def plot_error_per_class(all_labels, all_preds, classes):
    cm = confusion_matrix(all_labels, all_preds)
    # Tính False Negatives (bỏ lỡ) và False Positives (đoán sai)
    fn = np.sum(cm, axis=1) - np.diag(cm)
    fp = np.sum(cm, axis=0) - np.diag(cm)

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, fn, width, label='Missed (False Negative)', color='#ff7f7f')
    ax.bar(x + width / 2, fp, width, label='Wrongly Predicted (False Positive)', color='#7fbfff')

    ax.set_ylabel('Number of Critical Errors')
    ax.set_title('Analysis of Prediction Errors per Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/error_per_class.png", dpi=300)
    plt.show()

# ========================
# DATALOADER
# ========================
def get_dataloaders(train_data, test_data, transform, batch_size=BATCH_SIZE):
    full_train = CIFAR10Dataset(train_data, transform)
    test_dataset = CIFAR10Dataset(test_data, transform)

    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_subset, val_subset = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, name, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_acc = 0
    best_preds, best_labels = [], []
    corresponding_micro_f1 = 0
    corresponding_macro_f1 = 0
    total_train_time = 0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        print(f"\n{name} - Epoch {epoch + 1}/{EPOCHS}")
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        epoch_duration = time.time() - start_time  # Tính thời gian 1 epoch
        total_train_time += epoch_duration
        v_loss, acc, f1_micro, f1_macro, all_labels, all_preds = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {v_loss:.4f} | Acc: {acc:.4f} | Micro-F1: {f1_micro:.4f} "
              f"| Macro-F1: {f1_macro:.4f} | Time: {epoch_duration:.1f}s")

        if acc > best_acc:
            best_acc = acc
            best_preds = all_preds
            best_labels = all_labels
            corresponding_macro_f1 = f1_macro
            corresponding_micro_f1 = f1_micro
            torch.save(model.state_dict(), f"models/{name}_best.pth")
    avg_time = total_train_time / EPOCHS
    print(f"\n{name} Training Complete. Average Time per Epoch: {avg_time:.2f}s")
    return best_acc, corresponding_macro_f1, corresponding_micro_f1

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    dataset = load_dataset("uoft-cs/cifar10")
    t_cnn, t_vit = get_transforms()
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    train_data = dataset["train"]
    test_data = dataset["test"]

    # --- Training ResNet50 ---
    print("\n=== Training ResNet50 ===")
    train_l, val_l, test_l = get_dataloaders(train_data, test_data, t_cnn)
    resnet = get_model("resnet50").to(device)
    res_acc1, f1_macro_res, f1_micro_res = train_model(resnet, train_l, val_l, "resnet50", lr=1e-4)
    print(f"\nFinal ResNet: \nAccuracy {res_acc1:.4f}, \nF1_Micro {f1_micro_res:.4f}, \nF1_Macro {f1_macro_res:.4f}")

    # --- Training ViT ---
    print("\n=== Training ViT-Base (State-of-the-Art) ===")
    train_l_vit, val_l_vit, test_l_vit = get_dataloaders(train_data, test_data, t_vit)
    vit = get_model("vit").to(device)
    vit_acc1, f1_macro_vit, f1_micro_vit = train_model(vit, train_l_vit, val_l_vit, "vit", lr=5e-5)
    print(f"\nFinal ViT: \nAccuracy {vit_acc1:.4f}, \nF1_Micro {f1_micro_vit:.4f}, \nF1_Macro {f1_macro_vit:.4f}")
