"""
dog_dataset_manager.py
======================
Production-grade ML Dataset Manager for Dog Breed Images.
Uses a self-balancing-aware Binary Search Tree (BST) backed by
JSON persistence and an interactive CLI menu.

Author : Generated for production use
Python : 3.9+
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SNAPSHOT_FILE = "dataset_snapshot.json"
CORRUPTED_FILES = {".ds_store", "thumbs.db", "desktop.ini", ".localized"}
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}


# ===========================================================================
# 1.  BINARY SEARCH TREE IMPLEMENTATION
# ===========================================================================

class Node:
    """
    A single node in the DogDatasetBST.

    Attributes
    ----------
    breed_name  : str            – Primary key; used for alphabetic ordering.
    images_list : list[str]      – Relative paths / filenames for this breed.
    metadata    : dict           – count, date_added, status.
    left        : Node | None    – Left child (breeds that sort lower).
    right       : Node | None    – Right child (breeds that sort higher).
    height      : int            – Height of this subtree; used for AVL balancing.
    """

    __slots__ = ("breed_name", "images_list", "metadata", "left", "right", "height")

    def __init__(self, breed_name: str, image: Optional[str] = None) -> None:
        self.breed_name: str = breed_name
        self.images_list: list[str] = [image] if image else []
        self.metadata: dict = {
            "count": 1 if image else 0,
            "date_added": datetime.now().isoformat(timespec="seconds"),
            "status": "active",
        }
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.height: int = 1  # AVL height attribute


class DogDatasetBST:
    """
    AVL-balanced Binary Search Tree keyed on dog breed names.

    AVL invariant  : |height(left) - height(right)| ≤ 1 at every node.
    Insert         : O(log n) amortised.
    Search         : O(log n) amortised.
    In-order walk  : O(n).

    The tree automatically rebalances after every insertion via single and
    double (zig-zag) rotations, preventing worst-case O(n) degeneration that
    occurs when breeds are inserted in alphabetical order into a plain BST.
    """

    def __init__(self) -> None:
        self._root: Optional[Node] = None
        self._size: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, breed: str, image: Optional[str] = None) -> None:
        """Insert a breed / image pair into the tree.

        If the breed already exists its images_list is extended and the
        metadata count is incremented; no duplicate node is created.
        """
        breed = breed.strip().lower()
        if image:
            image = image.strip()
        self._root = self._insert(self._root, breed, image)

    def search(self, breed: str) -> Optional[Node]:
        """Return the Node for *breed*, or None if absent. O(log n)."""
        breed = breed.strip().lower()
        return self._search(self._root, breed)

    def in_order_traversal(self) -> Generator[Node, None, None]:
        """Yield nodes in alphabetical order (left → root → right). O(n)."""
        yield from self._in_order(self._root)

    def size(self) -> int:
        """Return the number of distinct breed nodes in the tree."""
        return self._size

    # ------------------------------------------------------------------
    # AVL helper utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _height(node: Optional[Node]) -> int:
        return node.height if node else 0

    def _update_height(self, node: Node) -> None:
        node.height = 1 + max(self._height(node.left), self._height(node.right))

    def _balance_factor(self, node: Optional[Node]) -> int:
        if node is None:
            return 0
        return self._height(node.left) - self._height(node.right)

    # ------------------------------------------------------------------
    # AVL rotations
    # ------------------------------------------------------------------

    def _rotate_right(self, y: Node) -> Node:
        """Single right rotation around *y*."""
        x = y.left
        t2 = x.right
        x.right = y
        y.left = t2
        self._update_height(y)
        self._update_height(x)
        return x

    def _rotate_left(self, x: Node) -> Node:
        """Single left rotation around *x*."""
        y = x.right
        t2 = y.left
        y.left = x
        x.right = t2
        self._update_height(x)
        self._update_height(y)
        return y

    def _rebalance(self, node: Node) -> Node:
        """Apply the correct AVL rotation(s) if *node* is unbalanced."""
        self._update_height(node)
        bf = self._balance_factor(node)

        # Left-heavy
        if bf > 1:
            if self._balance_factor(node.left) < 0:          # Left-Right case
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)

        # Right-heavy
        if bf < -1:
            if self._balance_factor(node.right) > 0:         # Right-Left case
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    # ------------------------------------------------------------------
    # Recursive workers
    # ------------------------------------------------------------------

    def _insert(self, node: Optional[Node], breed: str, image: Optional[str]) -> Node:
        # Base case: empty slot → create new node
        if node is None:
            self._size += 1
            return Node(breed, image)

        if breed < node.breed_name:
            node.left = self._insert(node.left, breed, image)
        elif breed > node.breed_name:
            node.right = self._insert(node.right, breed, image)
        else:
            # Breed already exists → append image and update count
            if image and image not in node.images_list:
                node.images_list.append(image)
                node.metadata["count"] = len(node.images_list)
            return node  # No structural change; skip rebalance

        return self._rebalance(node)

    def _search(self, node: Optional[Node], breed: str) -> Optional[Node]:
        if node is None:
            return None
        if breed == node.breed_name:
            return node
        if breed < node.breed_name:
            return self._search(node.left, breed)
        return self._search(node.right, breed)

    def _in_order(self, node: Optional[Node]) -> Generator[Node, None, None]:
        if node is None:
            return
        yield from self._in_order(node.left)
        yield node
        yield from self._in_order(node.right)

    # ------------------------------------------------------------------
    # Serialisation helpers (used by the persistence layer)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the entire tree into a plain Python dictionary."""
        return self._node_to_dict(self._root)

    def _node_to_dict(self, node: Optional[Node]) -> dict | None:
        if node is None:
            return None
        return {
            "breed_name": node.breed_name,
            "images_list": node.images_list,
            "metadata": node.metadata,
            "left": self._node_to_dict(node.left),
            "right": self._node_to_dict(node.right),
        }

    def from_dict(self, data: dict | None) -> None:
        """Reconstruct the tree from a serialised dictionary (no re-balancing
        needed because data was stored in BST order and we rebuild directly)."""
        self._root = None
        self._size = 0
        self._root = self._dict_to_node(data)
        # Recalculate size
        self._size = sum(1 for _ in self.in_order_traversal())

    def _dict_to_node(self, data: dict | None) -> Optional[Node]:
        if data is None:
            return None
        node = Node.__new__(Node)
        node.breed_name = data["breed_name"]
        node.images_list = data["images_list"]
        node.metadata = data["metadata"]
        node.left = self._dict_to_node(data["left"])
        node.right = self._dict_to_node(data["right"])
        # Recalculate height for AVL integrity
        node.height = 1 + max(
            self._height(node.left), self._height(node.right)
        )
        return node


# ===========================================================================
# 2.  FILE SYSTEM INGESTION PIPELINE
# ===========================================================================

def scan_folder(root_path: str, tree: DogDatasetBST) -> tuple[int, int]:
    """
    Scan a directory tree where each *immediate* sub-folder represents a breed.

    Expected layout
    ---------------
    root_path/
        golden_retriever/
            img001.jpg
            img002.png
        beagle/
            photo_a.jpg

    Parameters
    ----------
    root_path : str           – Path to the dataset root directory.
    tree      : DogDatasetBST – Tree to populate.

    Returns
    -------
    (breeds_found, images_found) counts.
    """
    root = Path(root_path)

    # --- validate root ---
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    breeds_found = 0
    images_found = 0

    try:
        breed_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    except PermissionError as exc:
        raise PermissionError(f"Cannot read directory '{root}': {exc}") from exc

    if not breed_dirs:
        print(f"  [WARNING] No sub-directories found in '{root}'. "
              "Expected one sub-folder per breed.")
        return 0, 0

    for breed_dir in breed_dirs:
        # Clean the breed name: strip whitespace, lowercase, replace separators
        raw_name = breed_dir.name
        breed_name = raw_name.strip().lower().replace("-", "_").replace(" ", "_")

        if not breed_name:
            print(f"  [SKIP] Unnamed directory at '{breed_dir}'.")
            continue

        breed_images: list[str] = []

        try:
            for file_path in breed_dir.iterdir():
                # Skip directories
                if file_path.is_dir():
                    continue

                filename_lower = file_path.name.lower()

                # Filter corrupted / system files
                if filename_lower in CORRUPTED_FILES:
                    continue
                if filename_lower.startswith("."):
                    continue

                # Filter non-image extensions
                if file_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                    continue

                breed_images.append(str(file_path.relative_to(root)))

        except PermissionError:
            print(f"  [SKIP] Permission denied reading '{breed_dir}'.")
            continue

        if not breed_images:
            # Insert breed with zero images so it still appears in the tree
            tree.insert(breed_name)
            breeds_found += 1
            print(f"  [INFO] Breed '{breed_name}' has no valid images.")
            continue

        for img in breed_images:
            tree.insert(breed_name, img)

        breeds_found += 1
        images_found += len(breed_images)

    return breeds_found, images_found


# ===========================================================================
# 3.  DATA PERSISTENCE  —  JSON EXPORT & IMPORT
# ===========================================================================

def save_to_json(tree: DogDatasetBST, filepath: str = SNAPSHOT_FILE) -> None:
    """
    Serialise the entire BST to a pretty-printed JSON file.

    The JSON stores the raw tree topology (left/right pointers represented as
    nested objects) so the exact same tree shape can be restored without
    re-scanning the filesystem.

    Parameters
    ----------
    tree     : DogDatasetBST – Source tree.
    filepath : str           – Destination JSON file path.
    """
    snapshot = {
        "metadata": {
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "total_breeds": tree.size(),
            "total_images": sum(
                len(node.images_list) for node in tree.in_order_traversal()
            ),
        },
        "tree": tree.to_dict(),
    }

    dest = Path(filepath)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2, ensure_ascii=False)
        print(f"  [OK] Snapshot saved → '{dest.resolve()}'")
    except PermissionError as exc:
        print(f"  [ERROR] Cannot write to '{filepath}': {exc}")
    except OSError as exc:
        print(f"  [ERROR] Disk I/O error: {exc}")


def load_from_json(tree: DogDatasetBST, filepath: str = SNAPSHOT_FILE) -> None:
    """
    Deserialise a JSON snapshot back into *tree*, replacing any existing data.

    Parameters
    ----------
    tree     : DogDatasetBST – Target tree (will be cleared first).
    filepath : str           – Source JSON file path.
    """
    src = Path(filepath)

    if not src.exists():
        print(f"  [ERROR] Snapshot file not found: '{src}'")
        return
    if not src.is_file():
        print(f"  [ERROR] '{src}' is not a regular file.")
        return

    try:
        with src.open("r", encoding="utf-8") as fh:
            snapshot = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"  [ERROR] Malformed JSON in '{src}': {exc}")
        return
    except PermissionError as exc:
        print(f"  [ERROR] Cannot read '{src}': {exc}")
        return

    if "tree" not in snapshot:
        print("  [ERROR] Snapshot file is missing the 'tree' key. Aborted.")
        return

    meta = snapshot.get("metadata", {})
    tree.from_dict(snapshot["tree"])

    print(f"  [OK] Loaded snapshot exported at {meta.get('exported_at', 'unknown')}.")
    print(f"       Breeds: {meta.get('total_breeds', '?')}  |  "
          f"Images: {meta.get('total_images', '?')}")


# ===========================================================================
# 4.  INTERACTIVE CONSOLE MENU
# ===========================================================================

MENU = """
╔══════════════════════════════════════════════════════╗
║        🐕  Dog Breed Dataset Manager  🐕             ║
╠══════════════════════════════════════════════════════╣
║  [1]  Scan local folder → populate tree              ║
║  [2]  Search for a breed                             ║
║  [3]  List all breeds (alphabetical + image counts)  ║
║  [4]  Save current state to JSON                     ║
║  [5]  Load state from JSON                           ║
║  [6]  Exit                                           ║
╚══════════════════════════════════════════════════════╝
"""


def _prompt(prompt_text: str) -> str:
    """Read a stripped string from stdin; handle EOF gracefully."""
    try:
        return input(prompt_text).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  [Interrupted]")
        return ""


def menu_scan(tree: DogDatasetBST) -> None:
    path = _prompt("  Enter the path to the dataset root folder: ")
    if not path:
        print("  [SKIP] No path entered.")
        return
    try:
        breeds, images = scan_folder(path, tree)
        print(f"\n  ✔ Scan complete: {breeds} breed(s), {images} image(s) loaded.")
    except FileNotFoundError as exc:
        print(f"  [ERROR] {exc}")
    except NotADirectoryError as exc:
        print(f"  [ERROR] {exc}")
    except PermissionError as exc:
        print(f"  [ERROR] {exc}")


def menu_search(tree: DogDatasetBST) -> None:
    query = _prompt("  Enter breed name to search: ")
    if not query:
        return
    node = tree.search(query)
    if node is None:
        print(f"  [NOT FOUND] No data for breed '{query}'.")
        return

    print(f"\n  ┌─ Breed : {node.breed_name}")
    print(f"  ├─ Count : {node.metadata['count']}")
    print(f"  ├─ Added : {node.metadata['date_added']}")
    print(f"  ├─ Status: {node.metadata['status']}")
    print(f"  └─ Images ({len(node.images_list)}):")
    for img in node.images_list[:20]:            # cap display at 20
        print(f"       • {img}")
    if len(node.images_list) > 20:
        print(f"       … and {len(node.images_list) - 20} more.")


def menu_list(tree: DogDatasetBST) -> None:
    if tree.size() == 0:
        print("  [INFO] The tree is empty. Scan a folder or load a snapshot first.")
        return

    print(f"\n  {'BREED':<35} {'IMAGES':>8}  STATUS")
    print("  " + "-" * 52)
    for node in tree.in_order_traversal():
        print(f"  {node.breed_name:<35} {node.metadata['count']:>8}  {node.metadata['status']}")
    print(f"\n  Total breeds: {tree.size()}")


def menu_save(tree: DogDatasetBST) -> None:
    path = _prompt(f"  Save path [{SNAPSHOT_FILE}]: ") or SNAPSHOT_FILE
    save_to_json(tree, path)


def menu_load(tree: DogDatasetBST) -> None:
    path = _prompt(f"  Load path [{SNAPSHOT_FILE}]: ") or SNAPSHOT_FILE
    load_from_json(tree, path)


def run_menu() -> None:
    """Main event loop for the interactive CLI."""
    tree = DogDatasetBST()

    dispatch = {
        "1": menu_scan,
        "2": menu_search,
        "3": menu_list,
        "4": menu_save,
        "5": menu_load,
    }

    while True:
        print(MENU)
        choice = _prompt("  Select an option [1-6]: ")

        if choice == "6":
            print("\n  Goodbye! 🐾\n")
            sys.exit(0)

        handler = dispatch.get(choice)
        if handler is None:
            print("  [INVALID] Please enter a number between 1 and 6.")
            continue

        handler(tree)
        print()     # breathing room between actions


# ===========================================================================
# 5.  ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    run_menu()
