# 🐕 Dog Pattern Recognition System

A web application that identifies dog breeds from uploaded photos using a fine-tuned ResNet-18 neural network, backed by custom data structures — a Binary Search Tree, a Max Heap, and a Queue — built from scratch in Python.

---

## What It Does

Upload any dog photo and the app returns the top 3 predicted breeds with confidence percentages. The web dashboard also shows a live breed leaderboard and a searchable breed catalogue backed by the BST.

---

## How to Run

**Step 1 — Install dependencies**
```bash
pip install flask torch torchvision pillow matplotlib scikit-learn
```

**Step 2 — Organise your images**

Put your dog photos in a folder where each subfolder is a breed name:
```
dog_images/
  golden_retriever/
    img001.jpg
    img002.jpg
  beagle/
    photo_a.jpg
```

**Step 3 — Build the dataset index**
```bash
python dog_dataset_manager.py
```
Choose option 1 to scan, then option 4 to save. This creates `dataset_snapshot.json`.

**Step 4 — Train the model**
```bash
python trainer.py
```
This fine-tunes ResNet-18 on your dataset and saves `dog_model.pth`. Takes a few minutes depending on dataset size.

**Step 5 — Launch the web app**
```bash
python app.py
```
Open your browser at **http://localhost:5000**

---

## The AI — How It Works

### Transfer Learning with ResNet-18

The recognition engine is built on **ResNet-18**, a convolutional neural network developed by Microsoft Research and pre-trained on ImageNet — a dataset of 1.2 million photos across 1,000 categories. ResNet-18 already knows how to detect edges, textures, fur patterns, body shapes, and facial structures from that prior training.

Rather than training a neural network from scratch — which would require millions of images and days of computation — this project uses **transfer learning**. The pre-trained ResNet-18 is loaded with its weights frozen, and only the final classification layer is replaced and trained from scratch:

```
Original ResNet-18 final layer:
  512 inputs → 1000 outputs (ImageNet categories)

Replaced with:
  Dropout(0.3) → Linear(512→256) → ReLU → Linear(256→7 breeds)
```

Freezing the earlier layers means the model keeps all its visual knowledge from ImageNet. Only the new final layers learn — making training fast and effective even with a small dataset.

### What the Network Actually Learns

ResNet-18 uses **residual connections** — shortcuts that let gradients flow directly through the network without degrading. This allows it to be 18 layers deep without the vanishing gradient problem that plagued earlier deep networks. Each layer learns increasingly abstract features:

- **Early layers** — edges, corners, basic textures
- **Middle layers** — fur patterns, ear shapes, eye structures
- **Deep layers** — full facial features, body proportions, breed-specific markings
- **Final layer (ours)** — maps those features to one of 7 breed categories

### Training Process

Training runs for **10 epochs** — meaning the model sees the entire dataset 10 times. Each pass through the data:

1. A batch of 16 images is fed to the model
2. The model makes predictions for each image
3. **Cross-entropy loss** calculates how wrong the predictions were — a single number measuring the gap between predictions and correct answers
4. **Backpropagation** traces that error backwards through the final layers, calculating how much each weight contributed
5. **Adam optimiser** adjusts the weights slightly in the direction that reduces the error
6. Repeat for the next batch

After every epoch, the model is tested on the 20% of images it was never trained on (the validation set). The best-performing checkpoint is saved — so even if later epochs overfit slightly, the saved model always reflects peak accuracy.

A **learning rate scheduler** watches the validation accuracy. If it stops improving for 2 consecutive epochs, the learning rate is halved automatically — taking smaller, more careful steps when big steps are no longer helping.

### Data Augmentation

Before each training image is shown to the model, it goes through random transformations:

- **Random horizontal flip** (50% chance) — mirrors the image
- **Random rotation** (up to ±15 degrees) — tilts the image slightly
- **Colour jitter** — varies brightness and contrast slightly

These aren't errors — they're deliberate. If the model only ever saw dogs standing perfectly upright under consistent lighting, it would fail on real-world photos. Augmentation forces it to learn what "dog" means regardless of orientation or lighting conditions.

Validation images are never augmented — they're kept clean so the accuracy measurement is honest.

### Making a Prediction

When a photo is uploaded:

1. Image is resized to **224×224 pixels** (ResNet-18's required input size)
2. Pixel values are normalised using ImageNet's mean and standard deviation — the same values used during the original pre-training, because the model's internal filters were calibrated for that specific numerical range
3. The tensor is passed through all 18 layers of ResNet-18
4. The output is 7 raw scores (called **logits**), one per breed
5. **Softmax** converts the logits into probabilities that sum to exactly 100%
6. The top 3 probabilities and their breed labels are returned

### Results

After training, `trainer.py` saves:
- `dog_model.pth` — the trained weights
- `results/training_curves.png` — accuracy and loss graphs across all epochs
- `results/classification_report.txt` — per-breed precision, recall, and F1 scores

---

## Data Structures

Three custom data structures are implemented from scratch. Each one solves a specific problem that the others cannot.

### Binary Search Tree (AVL-balanced)

**File:** `dog_dataset_manager.py`

The BST is the dataset catalogue. Every breed is one node storing the breed name, all image file paths, and metadata. Breeds are sorted alphabetically — left child comes before the current node, right child comes after.

Searching any breed takes **O(log n)** — about 7 steps for 100 breeds, 10 steps for 1000. Without balancing, inserting breeds in alphabetical order (as a folder scan produces) would create a straight line with O(n) search. AVL balancing prevents this by automatically rotating nodes after every insertion to keep both sides of the tree roughly equal in depth.

The BST connects to the web app at prediction time — after the model identifies a breed, `bst.search()` pulls that breed's image count to display alongside the confidence score.

### Max Heap

**File:** `app.py`

The Heap powers the breed leaderboard. It guarantees the breed with the most training images is always at index 0 — instantly accessible without sorting.

Stored as a flat Python list where parent-child relationships are determined by index math: parent of `i` is `(i-1)//2`, children are `2i+1` and `2i+2`. Insertions bubble up in O(log n), the maximum is retrieved in O(1). Populated at startup from the BST's in-order traversal.

### Queue (FIFO)

**File:** `app.py`

The Queue manages uploaded images. Every photo joins the back and is processed from the front — first in, first out. Backed by Python's `collections.deque` for O(1) operations at both ends. Maximum 10 images waiting at once. Processed images move to a history log keeping the last 10 results accessible.

---

## Project Structure

```
Dog-pattern-recognisiton/
├── app.py                  — Flask web app + Queue + Heap
├── dog_dataset_manager.py  — BST dataset manager + CLI
├── trainer.py              — ResNet-18 fine-tuning
├── recogniser.py           — Command-line breed recognition
├── templates/
│   └── index.html          — Web interface
├── dog_images/             — Training images (one folder per breed)
├── dataset_snapshot.json   — Auto-generated BST snapshot
└── dog_model.pth           — Auto-generated trained weights
```

> `dataset_snapshot.json` and `dog_model.pth` are auto-generated. Run `dog_dataset_manager.py` then `trainer.py` to create them.

---

## File Overview

| File | Purpose | Run |
|---|---|---|
| `dog_dataset_manager.py` | Scan images, build BST, save snapshot | Once |
| `trainer.py` | Fine-tune ResNet-18, save model weights | Once |
| `app.py` | Web interface, predictions, dashboard | Every time |
| `recogniser.py` | Command-line predictions | Optional |
