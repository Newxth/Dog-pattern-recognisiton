from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional


class Node:
    __slots__ = ("breed_name", "images_list", "metadata", "left", "right", "height")

    def __init__(self, breed_name: str, image: Optional[str] = None) -> None:
        self.breed_name = breed_name
        self.images_list = [image] if image else []
        self.metadata = {
            "count": 1 if image else 0,
            "date_added": datetime.now().isoformat(timespec="seconds"),
            "status": "active",
        }
        self.left = None
        self.right = None
        self.height = 1


class DogDatasetBST:
    def __init__(self) -> None:
        self._root = None
        self._size = 0

    def insert(self, breed: str, image: Optional[str] = None) -> None:
        breed = breed.strip().lower()
        if image:
            image = image.strip()
        self._root = self._insert(self._root, breed, image)

    def search(self, breed: str) -> Optional[Node]:
        breed = breed.strip().lower()
        return self._search(self._root, breed)

    def in_order_traversal(self) -> Generator[Node, None, None]:
        yield from self._in_order(self._root)

    def size(self) -> int:
        return self._size

    @staticmethod
    def _height(node):
        return node.height if node else 0

    def _update_height(self, node):
        node.height = 1 + max(self._height(node.left), self._height(node.right))

    def _balance_factor(self, node):
        if node is None:
            return 0
        return self._height(node.left) - self._height(node.right)

    def _rotate_right(self, y):
        x = y.left
        t2 = x.right
        x.right = y
        y.left = t2
        self._update_height(y)
        self._update_height(x)
        return x

    def _rotate_left(self, x):
        y = x.right
        t2 = y.left
        y.left = x
        x.right = t2
        self._update_height(x)
        self._update_height(y)
        return y

    def _rebalance(self, node):
        self._update_height(node)
        bf = self._balance_factor(node)

        if bf > 1:
            if self._balance_factor(node.left) < 0:
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)

        if bf < -1:
            if self._balance_factor(node.right) > 0:
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def _insert(self, node, breed, image):
        if node is None:
            self._size += 1
            return Node(breed, image)

        if breed < node.breed_name:
            node.left = self._insert(node.left, breed, image)
        elif breed > node.breed_name:
            node.right = self._insert(node.right, breed, image)
        else:
            if image and image not in node.images_list:
                node.images_list.append(image)
                node.metadata["count"] = len(node.images_list)
            return node

        return self._rebalance(node)

    def _search(self, node, breed):
        if node is None:
            return None
        if breed == node.breed_name:
            return node
        if breed < node.breed_name:
            return self._search(node.left, breed)
        return self._search(node.right, breed)

    def _in_order(self, node):
        if node is None:
            return
        yield from self._in_order(node.left)
        yield node
        yield from self._in_order(node.right)


def save_to_json(bst: DogDatasetBST, filepath: str) -> None:
    def node_to_dict(node):
        if node is None:
            return None
        return {
            "breed_name": node.breed_name,
            "images_list": node.images_list,
            "metadata": node.metadata,
            "left": node_to_dict(node.left),
            "right": node_to_dict(node.right),
        }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"tree": node_to_dict(bst._root), "size": bst.size()}, f, indent=2)


def load_from_json(bst: DogDatasetBST, filepath: str) -> None:
    def dict_to_node(d):
        if d is None:
            return None
        node = Node(d["breed_name"])
        node.images_list = d["images_list"]
        node.metadata = d["metadata"]
        node.left = dict_to_node(d.get("left"))
        node.right = dict_to_node(d.get("right"))
        node.height = 1 + max(
            node.left.height if node.left else 0,
            node.right.height if node.right else 0
        )
        return node

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    bst._root = dict_to_node(data.get("tree"))
    bst._size = data.get("size", 0)


if __name__ == "__main__":
    bst = DogDatasetBST()
    run_menu()