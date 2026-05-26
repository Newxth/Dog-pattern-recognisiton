"""
trainer.py
==========
Stage 2 — Model Training
Trains a ResNet18 neural network on your 7 dog breeds using
transfer learning. Saves the trained model and accuracy graphs.

Run AFTER dog_dataset_manager.py has scanned your dataset.
Python : 3.9+
Requires: torch, torchvision, pillow, matplotlib, scikit-learn
"""

import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# CONFIGURATION — change these if needed
# ---------------------------------------------------------------------------

DATASET_DIR   = "dog_images"          # folder with your 7 breed sub-folders
RESULTS_DIR   = "results"             # where graphs and stats are saved
MODEL_PATH    = "dog_model.pth"       # where the trained model is saved
SNAPSHOT_PATH = "dataset_snapshot.json"  # your BST snapshot

IMAGE_SIZE    = 224    # ResNet18 expects 224x224 pixels
BATCH_SIZE    = 16     # how many images to process at once
EPOCHS        = 10     # how many times to loop through all data
LEARNING_RATE = 0.001  # how fast the model adjusts (don't change)
TRAIN_SPLIT   = 0.8    # 80% training, 20% testing
RANDOM_SEED   = 42


# ---------------------------------------------------------------------------
# STEP 1 — LOAD BREED LABELS FROM YOUR BST SNAPSHOT
# ---------------------------------------------------------------------------

def load_breeds_from_snapshot(snapshot_path: str) -> list[str]:
    """
    Read breed names from your saved BST JSON snapshot.
    This connects Stage 1 (your BST) directly to Stage 2 (training).
    """
    path = Path(snapshot_path)
    if not path.exists():
        print(f"  [WARNING] Snapshot '{snapshot_path}' not found.")
        print("  Falling back to scanning dog_images folder directly.")
        return []

    with path.open("r", encoding="utf-8") as f:
        snapshot = json.load(f)

    # Walk the BST tree structure in the JSON to collect breed names
    breeds = []
    def collect_breeds(node):
        if node is None:
            return
        breeds.append(node["breed_name"])
        collect_breeds(node.get("left"))
        collect_breeds(node.get("right"))

    collect_breeds(snapshot.get("tree"))
    breeds.sort()
    return breeds


def get_breeds_from_folder(dataset_dir: str) -> list[str]:
    """Fallback: get breed names directly from sub-folder names."""
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: '{dataset_dir}'")
    breeds = sorted([
        d.name.strip().lower().replace("-", "_").replace(" ", "_")
        for d in root.iterdir() if d.is_dir()
    ])
    return breeds


# ---------------------------------------------------------------------------
# STEP 2 — CUSTOM DATASET CLASS
# ---------------------------------------------------------------------------

class DogBreedDataset(Dataset):
    """
    A PyTorch Dataset that reads dog images from your folder structure.

    PyTorch needs a Dataset object that can:
    - Tell it how many images exist  → __len__
    - Return one image + its label   → __getitem__
    """

    def __init__(self, dataset_dir: str, breeds: list[str], transform=None):
        self.transform  = transform
        self.breeds     = breeds
        self.breed_to_idx = {breed: idx for idx, breed in enumerate(breeds)}
        self.samples: list[tuple[Path, int]] = []

        root = Path(dataset_dir)

        # Map each folder to a breed index
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue

            # Clean folder name the same way the BST does
            clean_name = folder.name.strip().lower().replace("-", "_").replace(" ", "_")

            if clean_name not in self.breed_to_idx:
                continue

            label = self.breed_to_idx[clean_name]

            for img_path in folder.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((img_path, label))

        print(f"  Found {len(self.samples)} valid images across {len(breeds)} breeds.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # Return a blank image if the file is corrupted
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=0)

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# STEP 3 — IMAGE TRANSFORMS (pre-processing pipeline)
# ---------------------------------------------------------------------------

def get_transforms():
    """
    Two transform pipelines:
    - train_transform: includes random flips/rotations to help the model
      generalise (this is called Data Augmentation)
    - val_transform: just resize and normalise — no random changes
    """

    # These mean/std values are from ImageNet — required for ResNet18
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),      # randomly mirror image
        transforms.RandomRotation(degrees=15),        # randomly rotate ±15°
        transforms.ColorJitter(brightness=0.2,        # slightly vary colours
                               contrast=0.2),
        transforms.ToTensor(),                        # convert to numbers
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    return train_transform, val_transform


# ---------------------------------------------------------------------------
# STEP 4 — BUILD THE MODEL (Transfer Learning)
# ---------------------------------------------------------------------------

def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """
    Load a pretrained ResNet18 and replace its final layer
    with one that outputs num_classes predictions instead of 1000.

    ResNet18 was trained on ImageNet (1.2M images, 1000 categories).
    We borrow all its learned knowledge and just retrain the last layer
    for our 7 dog breeds. This is Transfer Learning.
    """
    # Load pretrained weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze all layers — we don't want to change what ResNet already learned
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classification layer
    # Original: 512 inputs → 1000 outputs (ImageNet categories)
    # Ours:     512 inputs → num_classes outputs (your 7 breeds)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),                    # prevents overfitting
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

    model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# STEP 5 — TRAINING LOOP
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, device, num_epochs):
    """
    The core training loop.

    Each epoch:
    1. Show the model every training image
    2. Model makes a prediction
    3. Calculate how wrong it was (loss)
    4. Adjust model weights to be less wrong next time
    5. Test on validation images to measure real accuracy
    """

    criterion = nn.CrossEntropyLoss()   # measures how wrong predictions are
    optimizer = optim.Adam(             # adjusts weights to reduce loss
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # Reduce learning rate if validation accuracy stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }

    best_val_acc  = 0.0
    best_model_state = None

    print(f"\n  {'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14} {'Val Loss':<14} {'Val Acc':<10}")
    print("  " + "─" * 60)

    for epoch in range(num_epochs):
        start = time.time()

        # ── Training phase ──────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()           # clear previous gradients
            outputs = model(images)         # forward pass
            loss = criterion(outputs, labels)
            loss.backward()                 # backward pass (backpropagation)
            optimizer.step()               # update weights

            train_loss    += loss.item() * images.size(0)
            _, predicted   = outputs.max(1)
            train_total   += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= train_total
        train_acc   = 100.0 * train_correct / train_total

        # ── Validation phase ─────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():   # don't track gradients during validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total   += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc   = 100.0 * val_correct / val_total

        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - start
        print(f"  {epoch+1:<8} {train_loss:<14.4f} {train_acc:<13.2f}% "
              f"{val_loss:<14.4f} {val_acc:<9.2f}% ({elapsed:.1f}s)")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    # Restore best weights
    if best_model_state:
        model.load_state_dict(best_model_state)

    print(f"\n  ✔ Best validation accuracy: {best_val_acc:.2f}%")
    return model, history


# ---------------------------------------------------------------------------
# STEP 6 — SAVE GRAPHS TO results/
# ---------------------------------------------------------------------------

def save_plots(history: dict, results_dir: str) -> None:
    """Save training accuracy and loss graphs as PNG files."""
    Path(results_dir).mkdir(exist_ok=True)
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy graph
    ax1.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    ax1.plot(epochs, history["val_acc"],   "r-o", label="Val Accuracy")
    ax1.set_title("Model Accuracy per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax1.grid(True)

    # Loss graph
    ax2.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax2.plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    ax2.set_title("Model Loss per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    out = Path(results_dir) / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [OK] Training curves saved → '{out}'")


def save_classification_report(model, val_loader, breeds, device, results_dir):
    """Save a per-breed accuracy breakdown."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds,
                                   target_names=breeds, digits=3)

    out = Path(results_dir) / "classification_report.txt"
    with open(out, "w") as f:
        f.write("DOG BREED CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)

    print(f"  [OK] Classification report saved → '{out}'")
    print("\n" + report)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  🐕  Dog Breed Trainer — Stage 2")
    print("=" * 60)

    # ── Device setup ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Using device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load breeds ─────────────────────────────────────────────────
    print(f"\n  Loading breeds from '{SNAPSHOT_PATH}'...")
    breeds = load_breeds_from_snapshot(SNAPSHOT_PATH)
    if not breeds:
        breeds = get_breeds_from_folder(DATASET_DIR)
    print(f"  Breeds found: {len(breeds)}")
    for i, b in enumerate(breeds):
        print(f"    [{i}] {b}")

    # ── Build datasets ───────────────────────────────────────────────
    print(f"\n  Loading images from '{DATASET_DIR}'...")
    train_tf, val_tf = get_transforms()

    full_dataset = DogBreedDataset(DATASET_DIR, breeds, transform=train_tf)

    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size   = len(full_dataset) - train_size

    torch.manual_seed(RANDOM_SEED)
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Apply clean transform to validation set
    val_set.dataset = DogBreedDataset(DATASET_DIR, breeds, transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"  Training images:   {train_size}")
    print(f"  Validation images: {val_size}")

    # ── Build model ──────────────────────────────────────────────────
    print(f"\n  Building ResNet18 model for {len(breeds)} breeds...")
    model = build_model(len(breeds), device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ── Train ────────────────────────────────────────────────────────
    print(f"\n  Starting training for {EPOCHS} epochs...")
    model, history = train_model(model, train_loader, val_loader,
                                 device, EPOCHS)

    # ── Save model ───────────────────────────────────────────────────
    torch.save({
        "model_state_dict": model.state_dict(),
        "breeds":           breeds,
        "num_classes":      len(breeds),
        "image_size":       IMAGE_SIZE,
    }, MODEL_PATH)
    print(f"\n  [OK] Model saved → '{MODEL_PATH}'")

    # ── Save results ─────────────────────────────────────────────────
    print(f"\n  Saving results to '{RESULTS_DIR}/'...")
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    save_plots(history, RESULTS_DIR)
    save_classification_report(model, val_loader, breeds, device, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("  ✅ Training complete! Run recogniser.py to test your model.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()