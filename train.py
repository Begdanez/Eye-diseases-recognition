import argparse
import os
import time
import copy
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B3_Weights

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CLASSES = ["AMD", "CNV", "CSR", "DME", "DR", "Drusen", "MH", "Normal"]
IMG_SIZE = 224          # EfficientNet-B3 default
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train retinal OCT classifier")
    parser.add_argument("--data_dir",    type=str, default="RetinalOCT_Dataset",
                        help="Root folder with train/val/test subfolders")
    parser.add_argument("--output_dir",  type=str, default="checkpoints",
                        help="Where to save model weights and metrics")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--workers",     type=int, default=4,
                        help="DataLoader num_workers")
    parser.add_argument("--freeze_epochs", type=int, default=5,
                        help="Epochs to train only the classifier head (backbone frozen)")
    parser.add_argument("--patience",    type=int, default=7,
                        help="Early-stopping patience (val loss not improving)")
    return parser.parse_args()


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf


def get_dataloaders(data_dir, batch_size, workers):
    train_tf, val_tf = get_transforms()

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=val_tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)

    print(f"  Train : {len(train_ds):,} images")
    print(f"  Val   : {len(val_ds):,}   images")
    print(f"  Test  : {len(test_ds):,}   images")
    print(f"  Classes: {train_ds.classes}")
    return train_loader, val_loader, test_loader, train_ds.classes


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """EfficientNet-B3 with custom classifier head."""
    model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model.to(device)


def freeze_backbone(model: nn.Module):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  Retinal OCT-C8  |  Device: {device}")
    print(f"{'='*50}\n")

    # ── Data ─────────────────────────────────
    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        args.data_dir, args.batch_size, args.workers
    )

    # ── Model ────────────────────────────────
    print("\nBuilding EfficientNet-B3 model...")
    model = build_model(len(class_names), device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training ─────────────────────────────
    best_val_acc  = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    patience_counter = 0
    best_val_loss = float("inf")

    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Phase 1: freeze backbone for first N epochs
        if epoch == 1:
            print(f"  [Phase 1] Backbone FROZEN for {args.freeze_epochs} epochs")
            freeze_backbone(model)
        if epoch == args.freeze_epochs + 1:
            print(f"\n  [Phase 2] Backbone UNFROZEN — full fine-tuning\n")
            unfreeze_backbone(model)
            # Lower LR for fine-tuning
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * 0.1

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:03d}/{args.epochs}]  "
              f"train loss={train_loss:.4f}  acc={train_acc:.4f}  |  "
              f"val loss={val_loss:.4f}  acc={val_acc:.4f}  "
              f"({elapsed:.1f}s)")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts,
                       os.path.join(args.output_dir, "best_model.pth"))
            print(f"  ✓ Best model saved  (val_acc={best_val_acc:.4f})")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    # ── Test evaluation ───────────────────────
    print("\n" + "="*50)
    print("Final evaluation on TEST set...")
    model.load_state_dict(best_model_wts)
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f"\n  Test Loss : {test_loss:.4f}")
    print(f"  Test Acc  : {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names))

    # ── Save artifacts ────────────────────────
    # Save class names mapping
    class_map = {i: name for i, name in enumerate(class_names)}
    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump(class_map, f, indent=2)

    # Save training history
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save confusion matrix
    cm = confusion_matrix(labels, preds)
    np.save(os.path.join(args.output_dir, "confusion_matrix.npy"), cm)

    print(f"\nAll artifacts saved to: {args.output_dir}/")
    print(f"  • best_model.pth")
    print(f"  • class_names.json")
    print(f"  • history.json")
    print(f"  • confusion_matrix.npy")
    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
