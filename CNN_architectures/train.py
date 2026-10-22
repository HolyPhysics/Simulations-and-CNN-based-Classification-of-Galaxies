"""
Train a ResNet-50 on the galaxy images (Spiral / Elliptical / Irregular).

Put this file next to Dataset.py (the file containing GalaxyCNNDataSet) and run:
    python train.py

Before running, edit Dataset.py:
  1. In __getitem__, use:  image_itself = Image.open(image_path).convert("RGB")
  2. Remove the print() calls inside __len__ and __getitem__.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet50, ResNet50_Weights

from Dataset import GalaxyCNNDataSet

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
IMAGE_DIR = "/Volumes/Research/emcgrath/Chidiebere_Okafor_N/Summer_Research_2026/latest_rgb_training_data/Images"
LABEL_DIR = "/Volumes/Research/emcgrath/Chidiebere_Okafor_N/Summer_Research_2026/latest_rgb_training_data/Labels"
OUTPUT_DIR = "outputs"

CLASS_NAMES = ["Spiral", "Elliptical", "Irregular"]   # indices 0, 1, 2
VAL_FRACTION = 0.2
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42

EPOCHS_HEAD = 5          # phase 1: only the new final layer trains
LR_HEAD = 1e-3
EPOCHS_FINETUNE = 20     # phase 2: whole network trains
LR_FINETUNE = 1e-4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def get_transforms():
    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),          # galaxies have no "up"
        T.RandomRotation(30),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = T.Compose([
        T.Resize(256, antialias=True),
        T.CenterCrop(224),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, val_transform


def read_valid_labels(dataset):
    """Return (indices, labels) for every image whose label is not 'Uncertain' (-1)."""
    indices, labels = [], []
    for i, file_name in enumerate(dataset.sorted_image_file_names):
        root, _ = os.path.splitext(file_name)
        label_path = os.path.join(dataset.label_directory, f"{root}.txt")
        with open(label_path, "r") as f:
            label = dataset.image_label_to_number_map[f.read().strip()]
        if label != -1:
            indices.append(i)
            labels.append(label)
    return np.array(indices), np.array(labels)


def stratified_split(indices, labels, val_fraction, seed):
    """Split so each class keeps the same proportion in train and validation."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        cls_idx = indices[labels == cls]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_fraction)))
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])
    return sorted(train_idx), sorted(val_idx)


# ----------------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------------
def build_model(num_classes):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)      # ImageNet-pretrained
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head (2048 -> 3)
    return model


def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable


def freeze_batchnorm(model):
    """Keep BatchNorm running stats fixed while the backbone is frozen."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm            # rows = true class, columns = predicted class


def per_class_metrics(cm):
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1


# ----------------------------------------------------------------------------
# One pass over a DataLoader (training if an optimizer is given, else validation)
# ----------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, frozen_backbone=False):
    training = optimizer is not None
    model.train(training)
    if training and frozen_backbone:
        freeze_batchnorm(model)

    total_loss, n_seen = 0.0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            n_seen += images.size(0)
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / n_seen, (preds == labels).mean(), preds, labels


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # --- data: two dataset objects so train/val get different transforms ----
    train_transform, val_transform = get_transforms()
    train_full = GalaxyCNNDataSet(IMAGE_DIR, LABEL_DIR, transform=train_transform)
    val_full = GalaxyCNNDataSet(IMAGE_DIR, LABEL_DIR, transform=val_transform)

    indices, labels = read_valid_labels(train_full)          # drops 'Uncertain'
    train_idx, val_idx = stratified_split(indices, labels, VAL_FRACTION, SEED)

    label_lookup = dict(zip(indices, labels))
    train_labels = np.array([label_lookup[i] for i in train_idx])
    counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    print(f"Train: {len(train_idx)} images | Val: {len(val_idx)} images")
    print("Train class counts:", dict(zip(CLASS_NAMES, counts.tolist())), flush=True)

    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(Subset(val_full, val_idx), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            persistent_workers=NUM_WORKERS > 0)

    # --- model, loss (class-weighted for imbalance) -------------------------
    model = build_model(len(CLASS_NAMES)).to(device)
    class_weights = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)),
                                 dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0
    best_path = os.path.join(OUTPUT_DIR, "best_resnet50.pt")

    phases = [
        ("head-only", EPOCHS_HEAD, LR_HEAD, False),
        ("fine-tune", EPOCHS_FINETUNE, LR_FINETUNE, True),
    ]

    epoch_counter = 0
    for phase_name, n_epochs, lr, backbone_trainable in phases:
        set_backbone_trainable(model, backbone_trainable)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        print(f"\n=== Phase: {phase_name} ({n_epochs} epochs, lr={lr}) ===", flush=True)

        for _ in range(n_epochs):
            epoch_counter += 1
            t0 = time.time()
            tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, device,
                                              optimizer, frozen_backbone=not backbone_trainable)
            va_loss, va_acc, va_pred, va_true = run_epoch(model, val_loader, criterion, device)
            scheduler.step()

            cm = confusion_matrix(va_true, va_pred, len(CLASS_NAMES))
            _, _, f1 = per_class_metrics(cm)
            macro_f1 = f1.mean()

            history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc);   history["val_acc"].append(va_acc)
            history["val_f1"].append(macro_f1)

            saved = ""
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save(model.state_dict(), best_path)
                saved = "  <- saved best"

            print(f"Epoch {epoch_counter:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.3f} macro-F1 {macro_f1:.3f} | "
                  f"{time.time() - t0:.0f}s{saved}", flush=True)

    # --- final evaluation with the best checkpoint --------------------------
    model.load_state_dict(torch.load(best_path, map_location=device))
    _, final_acc, preds, true = run_epoch(model, val_loader, criterion, device)
    cm = confusion_matrix(true, preds, len(CLASS_NAMES))
    precision, recall, f1 = per_class_metrics(cm)

    print(f"\nBest model | val accuracy {final_acc:.3f} | macro-F1 {f1.mean():.3f}")
    print(f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<12}{precision[i]:>10.3f}{recall[i]:>10.3f}{f1[i]:>10.3f}")
    print("\nConfusion matrix (rows = true, cols = predicted):\n", cm)

    # --- plots ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train"); axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    axes[2].imshow(cm, cmap="Blues")
    axes[2].set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axes[2].set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axes[2].set_xlabel("predicted"); axes[2].set_ylabel("true"); axes[2].set_title("Confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[2].text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_summary.png"), dpi=150)
    plt.show()


if __name__ == "__main__":      # required on macOS because DataLoader workers use "spawn"
    main()