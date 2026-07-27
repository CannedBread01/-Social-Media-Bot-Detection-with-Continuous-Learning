import os

import torch
from torch.utils.data import IterableDataset


class ProcessedDataset(IterableDataset):
    def __init__(self, mode: str, root : str |None = None, dataset_name = None):
        if root is None: root = "datasets/ProcessedDatasets"

        if mode == "train": mode = "Train"
        elif mode == "test": mode = "Test"
        self.embed_path = os.path.join(root, f"{dataset_name}{mode}Embed.pt")
        self.label_path = os.path.join(root, f"{dataset_name}{mode}Label.pt")

        if not os.path.exists(self.label_path): raise FileNotFoundError(f"Label file at {self.label_path} does not exist")
        if not os.path.exists(self.embed_path): raise FileNotFoundError(f"Embed file at {self.embed_path} does not exist")

    def __iter__(self):
        labels = torch.load(self.label_path, map_location=torch.device("cpu"))
        embeddings = torch.load(self.embed_path, map_location=torch.device("cpu"))

        for i in range(0, labels.shape[0]):
            yield embeddings[i], labels[i]