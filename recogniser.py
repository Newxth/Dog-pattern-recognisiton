"""
recogniser.py
=============
Stage 3 — Breed Recognition
Point this at any dog photo and it tells you the breed
and how confident it is.

Run AFTER trainer.py has created dog_model.pth.
Python : 3.9+
Requires: torch, torchvision, pillow, matplotlib
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MODEL_PATH  = "dog_model.pth"   # trained model from trainer.py
IMAGE_SIZE  = 224               # must match trainer.py
TOP_K       = 3                 # show top 3 predictions


# ---------------------------------------------------------------------------
# STEP 1 — LOAD THE TRAINED MODEL
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: torch.device):
    """
    Load the saved model and breed labels.
    Returns the model ready for inference and the breeds list.
    """
    path = Path(model_path)
    if not path.exists():
        print(f"\n  [ERROR] Model file not found: '{model_path}'")
        print("  Please run trainer.py first to train the model.")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device)
    breeds      = checkpoint["breeds"]
    num_classes = checkpoint["num_classes"]

    # Rebuild the exact same architecture as trainer.py
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
    model.eval()   # set to evaluation mode — disables dropout

    return model, breeds


# ---------------------------------------------------------------------------
# STEP 2 — PREPARE IMAGE
# ---------------------------------------------------------------------------

def prepare_image(image_path: str) -> torch.Tensor:
    """
    Load an image from disk and transform it into a tensor
    the model can process.

    The same normalisation values as training MUST be used —
    otherwise the model receives input it wasn't trained on.
    """
    path = Path(image_path)
    if not path.exists():
        print(f"\n  [ERROR] Image not found: '{image_path}'")
        sys.exit(1)

    try:
        image = Image.open(path).convert("RGB")
    except UnidentifiedImageError:
        print(f"\n  [ERROR] Cannot open image: '{image_path}'")
        print("  Make sure it is a valid JPG, PNG, or WEBP file.")
        sys.exit(1)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet values — same as trainer
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Add batch dimension: [C, H, W] → [1, C, H, W]
    tensor = transform(image).unsqueeze(0)
    return tensor, image   # return original image too for display


# ---------------------------------------------------------------------------
# STEP 3 — RUN PREDICTION
# ---------------------------------------------------------------------------

def predict(model, tensor: torch.Tensor, breeds: list[str],
            device: torch.device, top_k: int = TOP_K):
    """
    Run the image through the model and return the top-k predictions.

    Returns a list of (breed_name, confidence_percentage) tuples.
    """
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)             # raw scores for each breed
        probs   = F.softmax(outputs, dim=1) # convert to probabilities (sum = 1)

    # Get top-k predictions
    top_probs, top_indices = probs.topk(min(top_k, len(breeds)), dim=1)

    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        breed      = breeds[idx.item()]
        confidence = prob.item() * 100
        results.append((breed, confidence))

    return results


# ---------------------------------------------------------------------------
# STEP 4 — DISPLAY RESULTS
# ---------------------------------------------------------------------------

def display_results(image_path: str, original_image, predictions: list,
                    save_output: bool = True) -> None:
    """
    Show the image alongside a bar chart of predictions.
    Optionally saves the result to results/prediction_result.png
    """
    breed_labels  = [p[0].replace("_", " ").title() for p in predictions]
    confidences   = [p[1] for p in predictions]
    colours       = ["#2ecc71" if i == 0 else "#3498db"
                     for i in range(len(predictions))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel — the dog photo
    ax1.imshow(original_image)
    ax1.axis("off")
    top_breed = breed_labels[0].split("-")[-1].strip()
    ax1.set_title(f"Input Image", fontsize=13, fontweight="bold")

    # Right panel — confidence bar chart
    bars = ax2.barh(breed_labels[::-1], confidences[::-1],
                    color=colours[::-1], edgecolor="white", height=0.5)
    ax2.set_xlabel("Confidence (%)", fontsize=11)
    ax2.set_title("Breed Predictions", fontsize=13, fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.grid(axis="x", alpha=0.3)

    # Add confidence labels on bars
    for bar, conf in zip(bars, confidences[::-1]):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{conf:.1f}%", va="center", fontsize=10, fontweight="bold")

    plt.suptitle(
        f"🐕  Predicted: {breed_labels[0].split(chr(45))[-1].strip()}  "
        f"({confidences[0]:.1f}% confident)",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if save_output:
        out = Path("results") / "prediction_result.png"
        Path("results").mkdir(exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  [OK] Result image saved → '{out}'")

    plt.show()


def print_results(predictions: list, image_path: str) -> None:
    """Print a clean text summary of predictions."""
    print(f"\n  ┌─ Image: {Path(image_path).name}")
    print(f"  │")
    for rank, (breed, confidence) in enumerate(predictions, 1):
        # Clean up the breed name for display
        clean = breed.split("_", 1)[-1].replace("_", " ").title()
        bar   = "█" * int(confidence / 5)
        medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else "  "
        print(f"  │  {medal} #{rank}  {clean:<30} {confidence:>6.2f}%  {bar}")
    print(f"  └─")


# ---------------------------------------------------------------------------
# INTERACTIVE MENU
# ---------------------------------------------------------------------------

def run_interactive(model, breeds, device):
    """Let the user test multiple images in a loop."""
    print("\n" + "=" * 60)
    print("  🐕  Dog Breed Recogniser — Stage 3")
    print("=" * 60)
    print(f"  Model loaded. Recognises {len(breeds)} breeds.")
    print(f"  Breeds: {', '.join(b.split('_', 1)[-1].replace('_', ' ').title() for b in breeds)}")

    while True:
        print("\n" + "-" * 60)
        print("  Enter the path to a dog image to recognise it.")
        print("  Type 'quit' to exit.\n")

        try:
            user_input = input("  Image path: ").strip().strip('"')
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye! 🐾")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye! 🐾\n")
            break

        if not user_input:
            continue

        print(f"\n  Analysing '{Path(user_input).name}'...")

        tensor, original_image = prepare_image(user_input)
        predictions = predict(model, tensor, breeds, device)

        print_results(predictions, user_input)
        display_results(user_input, original_image, predictions, save_output=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # ── Device setup ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Using device: {device.type.upper()}")

    # ── Load model ───────────────────────────────────────────────────
    print(f"  Loading model from '{MODEL_PATH}'...")
    model, breeds = load_model(MODEL_PATH, device)
    print(f"  Model ready. {len(breeds)} breeds loaded.")

    # ── Check if image path was passed as command-line argument ──────
    if len(sys.argv) > 1:
        image_path  = sys.argv[1]
        tensor, original_image = prepare_image(image_path)
        predictions = predict(model, tensor, breeds, device)
        print_results(predictions, image_path)
        display_results(image_path, original_image, predictions)
    else:
        # No argument — run interactive menu
        run_interactive(model, breeds, device)


if __name__ == "__main__":
    main()