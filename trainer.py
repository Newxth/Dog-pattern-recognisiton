import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

DATASET_DIR = "dog_images"
RESULTS_DIR = "results"
MODEL_PATH = "dog_model.pth"
SNAPSHOT_PATH = "dataset_snapshot.json"
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42


def load_breeds_from_snapshot(snapshot_path):
    path = Path(snapshot_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        snapshot = json.load(f)
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


def get_breeds_from_folder(dataset_dir):
    root = Path(dataset_dir)
    return sorted([
        d.name.strip().lower().replace("-", "_").replace(" ", "_")
        for d in root.iterdir() if d.is_dir()
    ])


class DogBreedDataset(Dataset):
    def __init__(self, dataset_dir, breeds, transform=None):
        self.transform = transform
        self.breeds = breeds
        self.breed_to_idx = {breed: idx for idx, breed in enumerate(breeds)}
        self.samples = []
        root = Path(dataset_dir)
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            clean_name = folder.name.strip().lower().replace("-", "_").replace(" ", "_")
            if clean_name not in self.breed_to_idx:
                continue
            label = self.breed_to_idx[clean_name]
            for img_path in folder.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=0)
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, val_transform


def build_model(num_classes, device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )
    return model.to(device)


def train_model(model, train_loader, val_loader, device, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    breeds = load_breeds_from_snapshot(SNAPSHOT_PATH) or get_breeds_from_folder(DATASET_DIR)
    train_tf, val_tf = get_transforms()
    full_dataset = DogBreedDataset(DATASET_DIR, breeds, transform=train_tf)
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    torch.manual_seed(RANDOM_SEED)
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    val_set.dataset = DogBreedDataset(DATASET_DIR, breeds, transform=val_tf)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = build_model(len(breeds), device)
    model, history = train_model(model, train_loader, val_loader, device, EPOCHS)
    torch.save({
        "model_state_dict": model.state_dict(),
        "breeds": breeds,
        "num_classes": len(breeds),
        "image_size": IMAGE_SIZE,
    }, MODEL_PATH)


if __name__ == "__main__":
    main()