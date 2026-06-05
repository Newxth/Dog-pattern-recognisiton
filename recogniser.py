import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

MODEL_PATH = "dog_model.pth"
IMAGE_SIZE = 224
TOP_K      = 3


def load_model(model_path, device):
    path = Path(model_path)
    if not path.exists():
        print(f"Model not found: '{model_path}'. Run trainer.py first.")
        sys.exit(1)

    checkpoint  = torch.load(model_path, map_location=device)
    breeds      = checkpoint["breeds"]
    num_classes = checkpoint["num_classes"]

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, breeds


def prepare_image(image_path):
    path = Path(image_path)
    if not path.exists():
        print(f"Image not found: '{image_path}'")
        sys.exit(1)

    try:
        image = Image.open(path).convert("RGB")
    except UnidentifiedImageError:
        print(f"Cannot open image: '{image_path}'")
        sys.exit(1)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor, image


def predict(model, tensor, breeds, device, top_k=TOP_K):
    tensor = tensor.to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = F.softmax(outputs, dim=1)

    top_probs, top_indices = probs.topk(min(top_k, len(breeds)), dim=1)
    return [(breeds[idx.item()], prob.item() * 100)
            for prob, idx in zip(top_probs[0], top_indices[0])]


def display_results(image_path, original_image, predictions, save_output=True):
    breed_labels = [p[0].replace("_", " ").title() for p in predictions]
    confidences  = [p[1] for p in predictions]
    colours      = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(predictions))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.imshow(original_image)
    ax1.axis("off")
    ax1.set_title("Input Image", fontsize=13, fontweight="bold")

    bars = ax2.barh(breed_labels[::-1], confidences[::-1],
                    color=colours[::-1], edgecolor="white", height=0.5)
    ax2.set_xlabel("Confidence (%)", fontsize=11)
    ax2.set_title("Breed Predictions", fontsize=13, fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.grid(axis="x", alpha=0.3)

    for bar, conf in zip(bars, confidences[::-1]):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{conf:.1f}%", va="center", fontsize=10, fontweight="bold")

    plt.suptitle(
        f"Predicted: {breed_labels[0].split(chr(45))[-1].strip()} ({confidences[0]:.1f}% confident)",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if save_output:
        out = Path("results") / "prediction_result.png"
        Path("results").mkdir(exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")

    plt.show()


def print_results(predictions, image_path):
    print(f"\n  Image: {Path(image_path).name}")
    for rank, (breed, confidence) in enumerate(predictions, 1):
        clean = breed.split("_", 1)[-1].replace("_", " ").title()
        bar   = "█" * int(confidence / 5)
        medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else "  "
        print(f"  {medal} #{rank}  {clean:<30} {confidence:>6.2f}%  {bar}")


def run_interactive(model, breeds, device):
    print(f"Model loaded. Recognises {len(breeds)} breeds.")
    print(f"Breeds: {', '.join(b.split('_',1)[-1].replace('_',' ').title() for b in breeds)}")

    while True:
        try:
            user_input = input("\n  Image path (or 'quit'): ").strip().strip('"')
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q") or not user_input:
            break

        tensor, original_image = prepare_image(user_input)
        predictions = predict(model, tensor, breeds, device)
        print_results(predictions, user_input)
        display_results(user_input, original_image, predictions)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, breeds = load_model(MODEL_PATH, device)

    if len(sys.argv) > 1:
        tensor, original_image = prepare_image(sys.argv[1])
        predictions = predict(model, tensor, breeds, device)
        print_results(predictions, sys.argv[1])
        display_results(sys.argv[1], original_image, predictions)
    else:
        run_interactive(model, breeds, device)


if __name__ == "__main__":
    main()