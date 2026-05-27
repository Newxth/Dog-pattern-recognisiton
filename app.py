"""
app.py
======
Web Interface — Dog Pattern Recognition System
A Flask web application that ties all three stages together:
- Stage 1: BST dataset manager
- Stage 2: Trained ResNet18 model
- Stage 3: Live breed recognition via web upload

Run with: python app.py
Then open: http://localhost:5000
"""

import base64
import io
import json
import sys
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, render_template, request
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Import your existing modules
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from dog_dataset_manager import DogDatasetBST, load_from_json

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_PATH    = "dog_model.pth"
SNAPSHOT_PATH = "dataset_snapshot.json"
IMAGE_SIZE    = 224
MAX_QUEUE     = 10

app = Flask(__name__)

# ---------------------------------------------------------------------------
# SECOND DATA STRUCTURE — Image Processing Queue
# ---------------------------------------------------------------------------

class ImageQueue:

    def __init__(self, max_size: int = MAX_QUEUE):
        self._queue   = deque()
        self._max     = max_size
        self._history = []          # keeps last 10 processed results

    def enqueue(self, image_data: dict) -> bool:
        """Add an image to the back of the queue. Returns False if full."""
        if len(self._queue) >= self._max:
            return False
        self._queue.append(image_data)
        return True

    def dequeue(self) -> dict | None:
        """Remove and return the front image. Returns None if empty."""
        if self.is_empty():
            return None
        item = self._queue.popleft()
        self._history.append(item)
        if len(self._history) > 10:
            self._history.pop(0)
        return item

    def peek(self) -> dict | None:
        """Look at the front item without removing it."""
        return self._queue[0] if self._queue else None

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def size(self) -> int:
        return len(self._queue)

    def history(self) -> list:
        return list(reversed(self._history))

    def to_list(self) -> list:
        return list(self._queue)


# ---------------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# THIRD DATA STRUCTURE — Max Heap (breed ranking by image count)
# ---------------------------------------------------------------------------

class BreedMaxHeap:


    def __init__(self):
        self._heap = []

    def _parent(self, i):   return (i - 1) // 2
    def _left(self, i):     return 2 * i + 1
    def _right(self, i):    return 2 * i + 2

    def insert(self, breed_name: str, count: int) -> None:

        self._heap.append({"breed": breed_name, "count": count})
        self._bubble_up(len(self._heap) - 1)

    def get_max(self):
        """Return the breed with most images. O(1) — always index 0."""
        return self._heap[0] if self._heap else None

    def extract_max(self):

        if not self._heap:
            return None
        if len(self._heap) == 1:
            return self._heap.pop()


        max_item = self._heap[0]
        self._heap[0] = self._heap.pop()
        self._bubble_down(0)
        return max_item

    def get_sorted(self) -> list:

        return sorted(self._heap, key=lambda x: x["count"], reverse=True)

    def size(self) -> int:
        return len(self._heap)

    def _bubble_up(self, i: int) -> None:

        while i > 0:
            parent = self._parent(i)
            if self._heap[i]["count"] > self._heap[parent]["count"]:
                # Swap with parent — move up
                self._heap[i], self._heap[parent] = \
                    self._heap[parent], self._heap[i]
                i = parent
            else:
                break   # heap rule satisfied — stop

    def _bubble_down(self, i: int) -> None:

        size = len(self._heap)
        while True:
            largest = i
            left    = self._left(i)
            right   = self._right(i)

            if left < size and \
               self._heap[left]["count"] > self._heap[largest]["count"]:
                largest = left

            if right < size and \
               self._heap[right]["count"] > self._heap[largest]["count"]:
                largest = right

            if largest != i:
                self._heap[i], self._heap[largest] = \
                    self._heap[largest], self._heap[i]
                i = largest
            else:
                break   # heap rule satisfied — stop
bst         = DogDatasetBST()
image_queue = ImageQueue()
breed_heap  = BreedMaxHeap()
model       = None
breeds      = []
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def load_model():
    global model, breeds
    path = Path(MODEL_PATH)
    if not path.exists():
        print(f"  [WARNING] Model not found at '{MODEL_PATH}'.")
        print("  Run trainer.py first. Recognition will be disabled.")
        return False

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    breeds     = checkpoint["breeds"]

    m = models.resnet18(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, checkpoint["num_classes"])
    )
    m.load_state_dict(checkpoint["model_state_dict"])
    m = m.to(device)
    m.eval()
    model = m
    print(f"  [OK] Model loaded. {len(breeds)} breeds ready.")
    return True


def load_bst():
    global bst, breed_heap
    bst       = DogDatasetBST()
    breed_heap = BreedMaxHeap()

    if Path(SNAPSHOT_PATH).exists():
        load_from_json(bst, SNAPSHOT_PATH)
        print(f"  [OK] BST loaded. {bst.size()} breeds.")

        # Populate heap from BST data
        for node in bst.in_order_traversal():
            breed_heap.insert(node.breed_name, node.metadata["count"])
        print(f"  [OK] Max Heap built. Top breed: {breed_heap.get_max()}")
    else:
        print(f"  [WARNING] No snapshot found.")




def process_image(pil_image: Image.Image) -> list[tuple[str, float]]:

    if model is None:
        return []

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = F.softmax(outputs, dim=1)

    top_probs, top_indices = probs.topk(min(3, len(breeds)), dim=1)
    return [(breeds[i.item()], p.item() * 100)
            for p, i in zip(top_probs[0], top_indices[0])]


def image_to_base64(pil_image: Image.Image) -> str:

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_bst_tree_data():

    def node_to_dict(node):
        if node is None:
            return None
        clean = node.breed_name.split("_", 1)[-1].replace("_", " ").title()
        return {
            "name":  clean,
            "key":   node.breed_name,
            "count": node.metadata["count"],
            "left":  node_to_dict(node.left),
            "right": node_to_dict(node.right),
        }
    return node_to_dict(bst._root)


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    stats = {
        "total_breeds":  bst.size(),
        "total_images":  sum(len(n.images_list) for n in bst.in_order_traversal()),
        "model_ready":   model is not None,
        "queue_size":    image_queue.size(),
        "device":        device.type.upper(),
        "breeds":        [n.breed_name.split("_",1)[-1].replace("_"," ").title()
                          for n in bst.in_order_traversal()],
    }
    return render_template("index.html", stats=stats)


@app.route("/api/recognise", methods=["POST"])
def api_recognise():
    """Upload an image → add to queue → process → return predictions."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        pil_image = Image.open(file.stream).convert("RGB")
    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image file"}), 400

    # Add to queue
    entry = {
        "filename":  file.filename,
        "thumbnail": image_to_base64(pil_image.resize((100, 100))),
    }
    image_queue.enqueue(entry)

    # Process immediately (dequeue and run model)
    item        = image_queue.dequeue()
    predictions = process_image(pil_image)

    if not predictions:
        return jsonify({"error": "Model not loaded. Run trainer.py first."}), 503

    results = []
    for breed, conf in predictions:
        clean = breed.split("_", 1)[-1].replace("_", " ").title()

        # Look up this breed in the BST — connecting Stage 1 to Stage 3
        node = bst.search(breed)
        dataset_count = node.metadata["count"] if node else 0

        results.append({
            "breed":         clean,
            "breed_key":     breed,
            "confidence":    round(conf, 2),
            "dataset_count": dataset_count,
        })

    return jsonify({
        "filename":    file.filename,
        "thumbnail":   item["thumbnail"],
        "predictions": results,
        "queue_size":  image_queue.size(),
    })


@app.route("/api/bst")
def api_bst():

    return jsonify({
        "tree":   get_bst_tree_data(),
        "breeds": [
            {
                "name":  n.breed_name.split("_",1)[-1].replace("_"," ").title(),
                "key":   n.breed_name,
                "count": n.metadata["count"],
            }
            for n in bst.in_order_traversal()
        ],
    })


@app.route("/api/search/<breed_key>")
def api_search(breed_key):

    node = bst.search(breed_key)
    if not node:
        return jsonify({"found": False}), 404
    return jsonify({
        "found":      True,
        "breed_name": node.breed_name,
        "count":      node.metadata["count"],
        "date_added": node.metadata["date_added"],
        "status":     node.metadata["status"],
        "images":     node.images_list[:10],
    })


@app.route("/api/queue")
def api_queue():
    """Return current queue state."""
    return jsonify({
        "queue":   image_queue.to_list(),
        "size":    image_queue.size(),
        "history": image_queue.history()[-5:],
    })

@app.route("/api/heap")
def api_heap():
    """Return heap data for visualisation."""
    return jsonify({
        "heap":     breed_heap.get_sorted(),
        "max":      breed_heap.get_max(),
        "size":     breed_heap.size(),
        "raw":      breed_heap._heap,   # raw array for visualisation
    })
@app.route("/api/stats")
def api_stats():
    """Return live dataset statistics."""
    breeds_data = []
    for node in bst.in_order_traversal():
        breeds_data.append({
            "name":  node.breed_name.split("_",1)[-1].replace("_"," ").title(),
            "count": node.metadata["count"],
        })
    return jsonify({
        "breeds":       breeds_data,
        "total_breeds": bst.size(),
        "total_images": sum(b["count"] for b in breeds_data),
        "model_ready":  model is not None,
        "queue_size":   image_queue.size(),
    })


# ---------------------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🐕  Dog Pattern Recognition — Web Interface")
    print("=" * 60)
    print(f"\n  Device: {device.type.upper()}")
    load_bst()
    load_model()
    print("\n  Starting server...")
    print("  Open your browser at: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)