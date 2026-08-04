import os

import torch
from torch.utils.data import IterableDataset


class ProcessedDataset(IterableDataset):
    """
    Provides the stored tensor files as an iterable dataset.
    Primarily used to run the feature vector pipeline only once and reuse the resulting vectors.
    """
    def __init__(self, mode: str, root : str |None = None, dataset_name = None, label_mapping = None):
        """
        :param mode: Dataset split to use ("train" or "test").
        :param root: the filepath of the dataset directory in which the dataset is stored.
        :param dataset_name: the name of the dataset to use
        :param label_mapping: List of labels to use for provided samples`
        """
        if root is None: root = "datasets/ProcessedDatasets"
        self.label_mapping = label_mapping
        # build file path to dataset files
        if mode == "train": mode = "Train"
        elif mode == "test": mode = "Test"
        self.embed_path = os.path.join(root, f"{dataset_name}{mode}Embed.pt")
        self.label_path = os.path.join(root, f"{dataset_name}{mode}Label.pt")

        # check if files exist
        if not os.path.exists(self.label_path): raise FileNotFoundError(f"Label file at {self.label_path} does not exist")
        if not os.path.exists(self.embed_path): raise FileNotFoundError(f"Embed file at {self.embed_path} does not exist")

    def __iter__(self):
        # load tensors from file and iterate over the samples
        labels = torch.load(self.label_path, map_location=torch.device("cpu"))
        embeddings = torch.load(self.embed_path, map_location=torch.device("cpu"))

        # handle label remapping
        label_mapping = {}
        label_set = labels.unique()
        if self.label_mapping is not None:
            for i, existing_label in enumerate(label_set):
                label_mapping[existing_label.item()] = self.label_mapping[i]
        else:
            for i, existing_label in enumerate(label_set):
                label_mapping[existing_label.item()] = existing_label.item()

        print("label_mapping", label_mapping)
        for i in range(0, labels.shape[0]):
            yield embeddings[i], label_mapping[labels[i].item()]