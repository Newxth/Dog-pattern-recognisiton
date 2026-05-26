[Dog_Pattern_Recognition_Documentation.md](https://github.com/user-attachments/files/28273547/Dog_Pattern_Recognition_Documentation.md)
# Dog Pattern Recognition System
## Technical Documentation

**Repository:** https://github.com/Newxth/Dog-pattern-recognisiton  
**Language:** Python 3.9+  
**Version:** 1.0

---

## 1. What the Application Does

The Dog Pattern Recognition System is a three-stage web application that identifies dog breeds from uploaded photographs. It combines a classical machine learning pipeline with a Flask web interface, allowing users to upload any dog image and receive the top three predicted breeds along with confidence percentages.

**Stage 1 — Dataset Management (`dog_dataset_manager.py`):** Scans a local folder of dog images organised by breed, builds an internal dataset index, and provides an interactive command-line menu to search, list, save, and reload that index.

**Stage 2 — Model Training (`trainer.py`):** Uses the indexed dataset to fine-tune a pre-trained ResNet-18 convolutional neural network (PyTorch) on the user's breed collection. The trained model is saved to `dog_model.pth`.

**Stage 3 — Web Interface (`app.py`):** Launches a Flask server at `http://localhost:5000`. Users upload an image through the browser; the image passes through an upload queue, is classified by the model, and the top predictions are returned instantly alongside dataset statistics for each breed.

---

## 2. Data Structures

The application uses three custom data structures, each chosen to solve a specific problem efficiently.

### 2.1 Binary Search Tree (BST) — AVL-balanced
**File:** `dog_dataset_manager.py`  
**Class:** `DogDatasetBST`

The BST is the core dataset index. Each **node** stores one dog breed as its key (`breed_name`), together with a list of image file paths (`images_list`) and metadata (image count, date added, status).

Breeds are ordered alphabetically: breeds that sort lower go to the left child, higher to the right. This means searching for any breed takes **O(log n)** time rather than scanning a flat list.

The tree is **AVL-balanced**, meaning after every insertion it automatically performs left or right rotations to keep the height difference between any two sibling subtrees at most 1. Without this, inserting breeds in alphabetical order (as a folder scan produces) would degenerate into a linked list with O(n) search. The balance factor is maintained via `_bubble_up` / `_bubble_down` helper methods.

An **in-order traversal** (left → root → right) produces all breeds in alphabetical order in O(n) time, used by the "List all breeds" menu option and by the web dashboard.

The entire tree is serialised to `dataset_snapshot.json` and can be reloaded without re-scanning the filesystem.

### 2.2 Queue (FIFO) — Image Processing Queue
**File:** `app.py`  
**Class:** `ImageQueue`

Every image uploaded through the web interface is placed into a queue before being processed. This ensures images are handled in the order they arrive (First-In, First-Out). The queue is backed by Python's `collections.deque`, which provides **O(1)** `enqueue` (append to back) and **O(1)** `dequeue` (remove from front) — significantly faster than a plain list, where removing the front element requires shifting all remaining items.

The queue holds up to 10 items. Processed images are moved to a short history buffer so recently classified results remain accessible via the `/api/queue` endpoint.

### 2.3 Max Heap — Breed Ranking
**File:** `app.py`  
**Class:** `BreedMaxHeap`

A max heap ranks all breeds by the number of images in the dataset, always keeping the most-represented breed at index 0. Internally it uses a flat Python list where the parent of node `i` is at `(i-1)//2` and its children at `2i+1` and `2i+2`.

- **Insert:** adds to the end, then `_bubble_up` swaps the item upward until the heap rule (parent ≥ children) is restored — **O(log n)**.
- **get_max:** reads index 0 — **O(1)**.
- **extract_max:** swaps root with the last item, removes the last, then `_bubble_down` restores order — **O(log n)**.

This powers the breed leaderboard visible in the web dashboard and is populated at startup from the BST's in-order traversal.

---

## 3. How to Run the Program

### Prerequisites

```
Python 3.9 or higher
pip
```

### Step 1 — Install dependencies

```bash
pip install flask torch torchvision pillow
```

### Step 2 — Prepare your dataset

Organise dog images in a folder where each sub-folder is a breed name:

```
dog_images/
  golden_retriever/
    img001.jpg
    img002.jpg
  beagle/
    photo_a.jpg
```

### Step 3 — Build the dataset index

```bash
python dog_dataset_manager.py
```

Choose option **[1]** to scan your folder, then **[4]** to save the snapshot. This creates `dataset_snapshot.json`.

### Step 4 — Train the model

```bash
python trainer.py
```

This fine-tunes ResNet-18 on your dataset and saves `dog_model.pth`. Training may take several minutes depending on dataset size and whether a GPU is available (CUDA is used automatically if detected).

### Step 5 — Launch the web application

```bash
python app.py
```

Open your browser at **http://localhost:5000**, upload any dog photo, and the system will return the top-3 predicted breeds with confidence scores.

---

### File Overview

| File | Purpose |
|---|---|
| `dog_dataset_manager.py` | BST dataset manager + CLI menu |
| `trainer.py` | ResNet-18 fine-tuning script |
| `app.py` | Flask web server + Queue + Heap |
| `dataset_snapshot.json` | Persisted BST (auto-generated) |
| `dog_model.pth` | Trained model weights (auto-generated) |
| `dog_images/` | Training images (one folder per breed) |
| `test_dogs/` | Images for quick manual testing |
| `templates/` | HTML templates for the web interface |
