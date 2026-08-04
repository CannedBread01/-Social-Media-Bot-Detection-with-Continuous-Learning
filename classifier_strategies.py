from __future__ import annotations

from typing import Protocol
import numpy as np
import torch
from torch import nn
import hdbscan

from label_remapper import LabelRemapper
from ClassificationHeads import ProgressiveNeuralNetworkClassifier, BaselineClassificationHead, ConfidenceClassifier, \
    OneAgainstRestSVM, ReplayBuffer
from pipeline_utils import train_classifier, multi_svm_training_loop, confidence_classifier_training_loop

from torch.utils.data import DataLoader, TensorDataset



class ClassifierStrategy(Protocol):
    def initialize(
        self,
        feature_dim: int,
        config: dict,
        device: torch.device,
    ) -> dict:
        ...

    def train_stage(
        self,
        state: dict,
        stage: dict,
    ) -> tuple[dict, dict]:
        ...

    def predict_labels(
        self,
        state: dict,
        features: np.ndarray,
        evaluation_context: dict,
    ) -> list[str]:
        ...


class CalibratedProgressiveClassifier(nn.Module):
    """
    Wraps the existing PNN and learns one scale/bias pair per output class.
    """

    def __init__(
        self,
        base_classifier: nn.Module,
        num_classes: int,
    ):
        super().__init__()

        self.base_classifier = base_classifier

        self.log_scales = nn.Parameter(
            torch.zeros(num_classes)
        )

        self.biases = nn.Parameter(
            torch.zeros(num_classes)
        )

    def forward(self, features):
        logits = self.base_classifier(features)

        scales = torch.exp(self.log_scales)

        return logits * scales + self.biases

    def expand_classifier(self, num_to_add: int):
        """
        Expands the underlying PNN and adds matching calibration values.
        """

        base_parameters = list(
            self.base_classifier.expand_classifier(
                num_to_add=num_to_add
            )
        )

        old_log_scales = self.log_scales.detach()
        old_biases = self.biases.detach()

        new_log_scales = torch.zeros(
            num_to_add,
            device=old_log_scales.device,
        )

        new_biases = torch.zeros(
            num_to_add,
            device=old_biases.device,
        )

        self.log_scales = nn.Parameter(
            torch.cat(
                [old_log_scales, new_log_scales],
                dim=0,
            )
        )

        self.biases = nn.Parameter(
            torch.cat(
                [old_biases, new_biases],
                dim=0,
            )
        )

        return base_parameters + [
            self.log_scales,
            self.biases,
        ]


def replay_to_arrays(
    replay_by_label: dict[str, np.ndarray],
    feature_dim: int,
) -> tuple[np.ndarray, list[str]]:
    all_features = []
    all_labels = []

    for label, features in replay_by_label.items():
        if len(features) == 0:
            continue

        all_features.append(features)
        all_labels.extend([label] * len(features))

    if not all_features:
        return (
            np.empty(
                (0, feature_dim),
                dtype=np.float32,
            ),
            [],
        )

    return (
        np.vstack(all_features).astype(np.float32),
        all_labels,
    )


def update_replay(
    replay_by_label: dict[str, np.ndarray],
    features: np.ndarray,
    labels: list[str],
    replay_per_class: int,
    random_seed: int,
):
    labels_array = np.asarray(
        labels,
        dtype=object,
    )

    rng = np.random.default_rng(random_seed)

    for label in sorted(set(labels)):
        class_features = features[
            labels_array == label
        ]

        if label in replay_by_label:
            candidates = np.vstack(
                [
                    replay_by_label[label],
                    class_features,
                ]
            )
        else:
            candidates = class_features

        if len(candidates) > replay_per_class:
            selected = rng.choice(
                len(candidates),
                size=replay_per_class,
                replace=False,
            )

            candidates = candidates[selected]

        replay_by_label[label] = candidates.astype(
            np.float32
        )


def create_balanced_training_set(
    current_features: np.ndarray,
    current_labels: list[str],
    replay_by_label: dict[str, np.ndarray],
    samples_per_class: int,
    random_seed: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Builds one balanced training set from:
    - replay examples from older classes
    - all current-task examples
    """

    rng = np.random.default_rng(random_seed)

    feature_pools: dict[str, list[np.ndarray]] = {}

    for label, features in replay_by_label.items():
        if len(features) > 0:
            feature_pools[label] = [
                np.asarray(
                    features,
                    dtype=np.float32,
                )
            ]

    current_labels_array = np.asarray(
        current_labels,
        dtype=object,
    )

    for label in sorted(set(current_labels)):
        class_features = current_features[
            current_labels_array == label
        ]

        if label not in feature_pools:
            feature_pools[label] = []

        feature_pools[label].append(class_features)

    balanced_features = []
    balanced_labels = []

    for label in sorted(feature_pools):
        available_features = np.vstack(
            feature_pools[label]
        )

        replace = len(available_features) < samples_per_class

        selected = rng.choice(
            len(available_features),
            size=samples_per_class,
            replace=replace,
        )

        balanced_features.append(
            available_features[selected]
        )

        balanced_labels.extend(
            [label] * samples_per_class
        )

    return (
        np.vstack(balanced_features).astype(
            np.float32
        ),
        balanced_labels,
    )


def detect_current_dominated_cluster(
    reference_features: np.ndarray,
    current_features: np.ndarray,
    min_cluster_size: int,
    current_fraction_threshold: float,
) -> bool:
    """
    HDBSCAN novelty diagnostic:
    returns True when a cluster is mostly made of incoming-task samples.
    """

    if len(reference_features) == 0:
        return False

    combined = np.vstack(
        [
            reference_features,
            current_features,
        ]
    )

    is_current = np.concatenate(
        [
            np.zeros(
                len(reference_features),
                dtype=bool,
            ),
            np.ones(
                len(current_features),
                dtype=bool,
            ),
        ]
    )

    effective_min_cluster_size = min(
        min_cluster_size,
        max(2, len(combined) // 4),
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=effective_min_cluster_size,
        min_samples=min(5, effective_min_cluster_size),
        prediction_data=True,
    )

    cluster_labels = clusterer.fit_predict(combined)

    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:
            continue

        members = cluster_labels == cluster_id

        cluster_size = int(members.sum())
        current_count = int(
            (members & is_current).sum()
        )

        if cluster_size == 0:
            continue

        current_fraction = current_count / cluster_size

        if (
            current_count >= effective_min_cluster_size
            and current_fraction
            >= current_fraction_threshold
        ):
            return True

    return False


class MulticlassPNNStrategy:
    """
    Original multiclass continual-learning strategy.

    Task 0:
        genuine_user + fake_followers

    Later tasks:
        introduce one new bot class at a time.

    Persistent state contains the PNN, calibration parameters,
    LabelRemapper, known labels, and replay buffer.
    """

    def initialize(
        self,
        feature_dim: int,
        config: dict,
        device: torch.device,
    ) -> dict:
        return {
            "feature_dim": feature_dim,
            "device": device,
            "config": config,
            "classifier": None,
            "criterion": nn.CrossEntropyLoss(),
            "label_remapper": LabelRemapper(),
            "known_native_labels": set(),
            "index_to_label": {},
            "replay_by_label": {},
        }

    def train_stage(
        self,
        state: dict,
        stage: dict,
    ) -> tuple[dict, dict]:
        task_index = stage["task_index"]
        task_name = stage["task_name"]

        X_train = stage["X_train"]
        y_train = [str(label) for label in stage["y_train"]]

        config = state["config"]
        device = state["device"]

        current_labels = set(y_train)

        new_labels = sorted(
            current_labels
            - state["known_native_labels"]
        )

        print("Current labels:", sorted(current_labels))
        print("Unseen labels:", new_labels)

        replay_features, _ = replay_to_arrays(
            replay_by_label=state["replay_by_label"],
            feature_dim=state["feature_dim"],
        )

        if task_index == 0:
            detected_new_cluster = False
        else:
            detected_new_cluster = (
                detect_current_dominated_cluster(
                    reference_features=replay_features,
                    current_features=X_train,
                    min_cluster_size=config[
                        "hdbscan_min_cluster_size"
                    ],
                    current_fraction_threshold=config[
                        "hdbscan_current_fraction"
                    ],
                )
            )

        print(
            "HDBSCAN novelty signal:",
            detected_new_cluster,
        )

        classifier = state["classifier"]
        label_remapper = state["label_remapper"]

        if task_index == 0:
            if len(new_labels) < 2:
                raise RuntimeError(
                    "Task 0 must contain at least two classes."
                )

            for label in new_labels:
                label_remapper.register(label)

                label_index = label_remapper.convert(
                    [label]
                )[0]

                state["index_to_label"][label_index] = label

            state["known_native_labels"].update(
                new_labels
            )

            base_classifier = (
                ProgressiveNeuralNetworkClassifier(
                    in_features=state["feature_dim"],
                    output_dim=len(new_labels),
                    dropout_p=config.get("dropout_p", 0.1),
                )
            )

            classifier = CalibratedProgressiveClassifier(
                base_classifier=base_classifier,
                num_classes=len(new_labels),
            ).to(device)

            trainable_parameters = list(
                classifier.parameters()
            )

            print(
                f"Created classifier with "
                f"{len(new_labels)} outputs."
            )

        else:
            use_override = config.get(
                "use_intervention_override",
                True,
            )

            should_expand = (
                len(new_labels) > 0
                if use_override
                else detected_new_cluster
            )

            if new_labels and not should_expand:
                raise RuntimeError(
                    "New labels exist, but the novelty "
                    "detector did not request expansion."
                )

            if should_expand and new_labels:
                for label in new_labels:
                    label_remapper.register(label)

                    label_index = label_remapper.convert(
                        [label]
                    )[0]

                    state["index_to_label"][label_index] = label

                state["known_native_labels"].update(
                    new_labels
                )

                trainable_parameters = list(
                    classifier.expand_classifier(
                        num_to_add=len(new_labels)
                    )
                )

                print(
                    f"Expanded classifier by "
                    f"{len(new_labels)} output(s): "
                    f"{new_labels}"
                )

            else:
                trainable_parameters = [
                    parameter
                    for parameter in classifier.parameters()
                    if parameter.requires_grad
                ]

        classifier.eval()

        with torch.inference_mode():
            probe_count = min(2, len(X_train))

            probe = torch.tensor(
                X_train[:probe_count],
                dtype=torch.float32,
                device=device,
            )

            actual_output_count = classifier(
                probe
            ).shape[1]

        expected_output_count = len(
            state["known_native_labels"]
        )

        if actual_output_count != expected_output_count:
            raise RuntimeError(
                f"Expected {expected_output_count} outputs, "
                f"but classifier has {actual_output_count}."
            )

        print(
            "Actual classifier outputs:",
            actual_output_count,
        )

        train_features, train_native_labels = (
            create_balanced_training_set(
                current_features=X_train,
                current_labels=y_train,
                replay_by_label=state["replay_by_label"],
                samples_per_class=config[
                    "balanced_samples_per_class"
                ],
                random_seed=config["random_seed"],
            )
        )

        print("Balanced classifier-training counts:")

        for label in sorted(set(train_native_labels)):
            print(
                f"  {label}: "
                f"{train_native_labels.count(label)}"
            )

        train_targets = label_remapper.convert(
            train_native_labels
        )

        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=config["learning_rate"],
        )

        classifier.train()

        train_classifier(
            classifier,
            state["criterion"],
            optimizer,
            train_features,
            train_targets,
            epochs=config["epochs"],
            device=device,
        )

        update_replay(
            replay_by_label=state["replay_by_label"],
            features=X_train,
            labels=y_train,
            replay_per_class=config[
                "replay_per_class"
            ],
            random_seed=config["random_seed"],
        )

        state["classifier"] = classifier

        train_info = {
            "task_name": task_name,
            "new_labels": new_labels,
            "hdbscan_detected_new_cluster": (
                detected_new_cluster
            ),
            "total_classes": expected_output_count,
        }

        return state, train_info

    def predict_labels(
        self,
        state: dict,
        features: np.ndarray,
        evaluation_context: dict,
    ) -> list[str]:
        classifier = state["classifier"]
        device = state["device"]

        classifier.eval()

        batch_size = state["config"].get(
            "eval_batch_size",
            512,
        )

        predicted_indices = []

        with torch.inference_mode():
            for start in range(
                0,
                len(features),
                batch_size,
            ):
                batch_features = torch.tensor(
                    features[start:start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )

                logits = classifier(batch_features)

                predicted_indices.extend(
                    logits.argmax(dim=1)
                    .cpu()
                    .tolist()
                )

        return [
            state["index_to_label"][index]
            for index in predicted_indices
        ]
class ProfessorSelfAttention(nn.Module):
    """
    Adapted from the professor's Transformer SelfAttention code.
    """

    def __init__(self, embed_size, heads):
        super().__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(
            self.head_dim,
            self.head_dim,
            bias=False,
        )

        self.keys = nn.Linear(
            self.head_dim,
            self.head_dim,
            bias=False,
        )

        self.queries = nn.Linear(
            self.head_dim,
            self.head_dim,
            bias=False,
        )

        self.fc_out = nn.Linear(
            heads * self.head_dim,
            embed_size,
        )

    def forward(self, values, keys, query, mask=None):
        batch_size = query.shape[0]

        value_len = values.shape[1]
        key_len = keys.shape[1]
        query_len = query.shape[1]

        values = values.reshape(
            batch_size,
            value_len,
            self.heads,
            self.head_dim,
        )

        keys = keys.reshape(
            batch_size,
            key_len,
            self.heads,
            self.head_dim,
        )

        queries = query.reshape(
            batch_size,
            query_len,
            self.heads,
            self.head_dim,
        )

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum(
            "nqhd,nkhd->nhqk",
            [queries, keys],
        )

        if mask is not None:
            energy = energy.masked_fill(
                mask == 0,
                float("-1e20"),
            )

        attention = torch.softmax(
            energy / (self.embed_size ** 0.5),
            dim=3,
        )

        out = torch.einsum(
            "nhql,nlhd->nqhd",
            [attention, values],
        ).reshape(
            batch_size,
            query_len,
            self.heads * self.head_dim,
        )

        return self.fc_out(out)


class ProfessorTransformerBlock(nn.Module):
    """
    Adapted from the professor's TransformerBlock code.
    """

    def __init__(
        self,
        embed_size,
        heads,
        dropout,
        forward_expansion,
    ):
        super().__init__()

        self.attention = ProfessorSelfAttention(
            embed_size,
            heads,
        )

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(
                embed_size,
                forward_expansion * embed_size,
            ),
            nn.ReLU(),
            nn.Linear(
                forward_expansion * embed_size,
                embed_size,
            ),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention = self.attention(
            value,
            key,
            query,
            mask,
        )

        x = self.dropout(
            self.norm1(attention + query)
        )

        forward = self.feed_forward(x)

        out = self.dropout(
            self.norm2(forward + x)
        )

        return out


class ProfessorFeatureTransformerClassifier(nn.Module):
    """
    Modified version of the professor's Transformer.

    Instead of source/target translation:
        token sequence -> translated token sequence

    We use:
        UMAP feature vector -> Transformer encoder -> class logits
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        embed_size: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        forward_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_size = embed_size

        # Each UMAP dimension becomes one position in a sequence.
        self.feature_projection = nn.Linear(
            1,
            embed_size,
        )

        self.position_embedding = nn.Embedding(
            input_dim,
            embed_size,
        )

        self.layers = nn.ModuleList(
            [
                ProfessorTransformerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier_head = nn.Linear(
            embed_size,
            num_classes,
        )

    def forward(self, features):
        """
        features shape:
            (batch_size, input_dim)

        Convert to:
            (batch_size, sequence_length=input_dim, 1)
        """

        batch_size, input_dim = features.shape

        if input_dim != self.input_dim:
            raise RuntimeError(
                f"Expected input_dim={self.input_dim}, "
                f"but got {input_dim}."
            )

        x = features.unsqueeze(-1)

        x = self.feature_projection(x)

        positions = torch.arange(
            0,
            input_dim,
            device=features.device,
        ).unsqueeze(0).expand(
            batch_size,
            input_dim,
        )

        x = self.dropout(
            x + self.position_embedding(positions)
        )

        for layer in self.layers:
            x = layer(x, x, x, mask=None)

        pooled = x.mean(dim=1)

        return self.classifier_head(pooled)

    def expand_classes(self, num_to_add: int):
        """
        Expands the final classifier head when new classes arrive.
        """

        old_head = self.classifier_head

        old_num_classes = old_head.out_features
        new_num_classes = old_num_classes + num_to_add

        new_head = nn.Linear(
            self.embed_size,
            new_num_classes,
        ).to(old_head.weight.device)

        with torch.no_grad():
            new_head.weight[:old_num_classes] = old_head.weight
            new_head.bias[:old_num_classes] = old_head.bias

        self.classifier_head = new_head


class MulticlassTransformerStrategy:
    """
    Continual-learning strategy using the professor's Transformer blocks.

    It is not PNN.
    It is a normal Transformer classifier with:
    - replay buffer
    - expanding output layer
    - same task sequence
    - same FeaturePipeline
    - same ExperimentRunner
    """

    def initialize(
        self,
        feature_dim: int,
        config: dict,
        device: torch.device,
    ) -> dict:
        return {
            "feature_dim": feature_dim,
            "device": device,
            "config": config,
            "classifier": None,
            "criterion": nn.CrossEntropyLoss(),
            "label_remapper": LabelRemapper(),
            "known_native_labels": set(),
            "index_to_label": {},
            "replay_by_label": {},
        }

    def train_stage(
        self,
        state: dict,
        stage: dict,
    ) -> tuple[dict, dict]:
        task_index = stage["task_index"]
        task_name = stage["task_name"]

        X_train = stage["X_train"]
        y_train = [str(label) for label in stage["y_train"]]

        config = state["config"]
        device = state["device"]

        current_labels = set(y_train)

        new_labels = sorted(
            current_labels
            - state["known_native_labels"]
        )

        print("Current labels:", sorted(current_labels))
        print("Unseen labels:", new_labels)

        label_remapper = state["label_remapper"]

        if task_index == 0:
            if len(new_labels) < 2:
                raise RuntimeError(
                    "Task 0 must contain at least two classes."
                )

            for label in new_labels:
                label_remapper.register(label)

                label_index = label_remapper.convert(
                    [label]
                )[0]

                state["index_to_label"][label_index] = label

            state["known_native_labels"].update(new_labels)

            classifier = ProfessorFeatureTransformerClassifier(
                input_dim=state["feature_dim"],
                num_classes=len(new_labels),
                embed_size=config.get(
                    "transformer_embed_size",
                    64,
                ),
                num_layers=config.get(
                    "transformer_num_layers",
                    2,
                ),
                heads=config.get(
                    "transformer_heads",
                    4,
                ),
                forward_expansion=config.get(
                    "transformer_forward_expansion",
                    4,
                ),
                dropout=config.get(
                    "dropout_p",
                    0.1,
                ),
            ).to(device)

            print(
                f"Created professor Transformer classifier "
                f"with {len(new_labels)} outputs."
            )

        else:
            classifier = state["classifier"]

            if new_labels:
                for label in new_labels:
                    label_remapper.register(label)

                    label_index = label_remapper.convert(
                        [label]
                    )[0]

                    state["index_to_label"][label_index] = label

                state["known_native_labels"].update(new_labels)

                classifier.expand_classes(
                    num_to_add=len(new_labels)
                )

                print(
                    f"Expanded Transformer classifier by "
                    f"{len(new_labels)} output(s): "
                    f"{new_labels}"
                )

        state["classifier"] = classifier

        classifier.eval()

        with torch.inference_mode():
            probe = torch.tensor(
                X_train[: min(2, len(X_train))],
                dtype=torch.float32,
                device=device,
            )

            actual_output_count = classifier(probe).shape[1]

        expected_output_count = len(
            state["known_native_labels"]
        )

        if actual_output_count != expected_output_count:
            raise RuntimeError(
                f"Expected {expected_output_count} outputs, "
                f"but classifier has {actual_output_count}."
            )

        print(
            "Actual Transformer classifier outputs:",
            actual_output_count,
        )

        train_features, train_native_labels = (
            create_balanced_training_set(
                current_features=X_train,
                current_labels=y_train,
                replay_by_label=state["replay_by_label"],
                samples_per_class=config[
                    "balanced_samples_per_class"
                ],
                random_seed=config["random_seed"],
            )
        )

        print("Balanced Transformer-training counts:")

        for label in sorted(set(train_native_labels)):
            print(
                f"  {label}: "
                f"{train_native_labels.count(label)}"
            )

        train_targets = label_remapper.convert(
            train_native_labels
        )

        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=config["learning_rate"],
        )

        train_classifier(
            classifier,
            state["criterion"],
            optimizer,
            train_features,
            train_targets,
            epochs=config["epochs"],
            device=device,
        )

        update_replay(
            replay_by_label=state["replay_by_label"],
            features=X_train,
            labels=y_train,
            replay_per_class=config[
                "replay_per_class"
            ],
            random_seed=config["random_seed"],
        )

        train_info = {
            "task_name": task_name,
            "new_labels": new_labels,
            "total_classes": expected_output_count,
            "model_type": "professor_transformer_classifier",
        }

        return state, train_info

    def predict_labels(
        self,
        state: dict,
        features: np.ndarray,
        evaluation_context: dict,
    ) -> list[str]:
        classifier = state["classifier"]
        device = state["device"]

        classifier.eval()

        batch_size = state["config"].get(
            "eval_batch_size",
            512,
        )

        predicted_indices = []

        with torch.inference_mode():
            for start in range(
                0,
                len(features),
                batch_size,
            ):
                batch_features = torch.tensor(
                    features[start:start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )

                logits = classifier(batch_features)

                predicted_indices.extend(
                    logits.argmax(dim=1)
                    .cpu()
                    .tolist()
                )

        return [
            state["index_to_label"][index]
            for index in predicted_indices
        ]


# -------------------
class BaselineClassifierStrategy:
    """
    """

    def initialize(self, feature_dim: int, config: dict, device: torch.device) -> dict:
        return {
            "feature_dim": feature_dim,
            "device": device,
            "config": config,
            "classifier": None,
            "criterion": nn.CrossEntropyLoss(),
            "label_remapper": LabelRemapper(),
            "known_native_labels": set(),
            "index_to_label": {},
            "replay_by_label": {},
        }

    def train_stage(self, state: dict, stage: dict) -> tuple[dict, dict]:
        task_index = stage["task_index"]
        task_name = stage["task_name"]

        X_train = stage["X_train"]
        y_train = [str(label) for label in stage["y_train"]]

        config = state["config"]
        device = state["device"]

        current_labels = set(y_train)

        new_labels = sorted(current_labels - state["known_native_labels"])

        print("Current labels:", sorted(current_labels))
        print("Unseen labels:", new_labels)

        replay_features, _ = replay_to_arrays(
            replay_by_label=state["replay_by_label"],
            feature_dim=state["feature_dim"],
        )

        if task_index == 0:
            detected_new_cluster = False
        else:
            detected_new_cluster = (
                detect_current_dominated_cluster(
                    reference_features=replay_features,
                    current_features=X_train,
                    min_cluster_size=config["hdbscan_min_cluster_size"],
                    current_fraction_threshold=config["hdbscan_current_fraction"],
                )
            )

        print("HDBSCAN novelty signal:", detected_new_cluster)

        classifier = state["classifier"]
        label_remapper = state["label_remapper"]

        if task_index == 0:
            if len(new_labels) < 2:
                raise RuntimeError("Task 0 must contain at least two classes.")

            for label in new_labels:
                label_remapper.register(label)
                label_index = label_remapper.convert([label])[0]
                state["index_to_label"][label_index] = label

            state["known_native_labels"].update(new_labels)

            base_classifier = BaselineClassificationHead(embedding_dim=state["feature_dim"], output_dim=8, dropout_prob=config.get("dropout_p", 0.1))
            classifier = base_classifier.to(device=device)

            trainable_parameters = list(
                classifier.parameters()
            )

            print(f"Created classifier with {len(new_labels)} outputs.")

        else:
            use_override = config.get("use_intervention_override",True)

            should_expand = (
                len(new_labels) > 0
                if use_override
                else detected_new_cluster
            )

            if new_labels and not should_expand:
                raise RuntimeError(
                    "New labels exist, but the novelty "
                    "detector did not request expansion."
                )

            if should_expand and new_labels:
                for label in new_labels:
                    label_remapper.register(label)

                    label_index = label_remapper.convert([label])[0]
                    state["index_to_label"][label_index] = label

                state["known_native_labels"].update(new_labels)

                trainable_parameters = list(
                    classifier.expand_classifier(num_to_add=len(new_labels))
                )

                print(f"Expanded classifier by {len(new_labels)} output(s): {new_labels}")

            else:
                trainable_parameters = [parameter for parameter in classifier.parameters() if
                                        parameter.requires_grad]

        classifier.eval()

        with torch.inference_mode():
            probe_count = min(2, len(X_train))

            probe = torch.tensor(X_train[:probe_count], dtype=torch.float32, device=device)

            actual_output_count = classifier(probe).shape[1]

        expected_output_count = len(state["known_native_labels"])


        print("Actual classifier outputs:", actual_output_count, )

        train_features, train_native_labels = (
            create_balanced_training_set(
                current_features=X_train,
                current_labels=y_train,
                replay_by_label=state["replay_by_label"],
                samples_per_class=config[
                    "balanced_samples_per_class"
                ],
                random_seed=config["random_seed"],
            )
        )

        print("Balanced classifier-training counts:")

        for label in sorted(set(train_native_labels)):
            print(f"  {label}: ", f"{train_native_labels.count(label)}")

        train_targets = label_remapper.convert(train_native_labels)

        optimizer = torch.optim.Adam(trainable_parameters, lr=config["learning_rate"], )

        classifier.train()

        train_classifier(
            classifier,
            state["criterion"],
            optimizer,
            train_features,
            train_targets,
            epochs=config["epochs"],
            device=device,
        )

        update_replay(
            replay_by_label=state["replay_by_label"],
            features=X_train,
            labels=y_train,
            replay_per_class=config["replay_per_class"],
            random_seed=config["random_seed"],
        )

        state["classifier"] = classifier

        train_info = {
            "task_name": task_name,
            "new_labels": new_labels,
            "hdbscan_detected_new_cluster": (detected_new_cluster),
            "total_classes": expected_output_count,
        }

        return state, train_info

    def predict_labels(self, state: dict, features: np.ndarray, evaluation_context: dict, ) -> list[str]:
        classifier = state["classifier"]
        device = state["device"]

        classifier.eval()

        batch_size = state["config"].get("eval_batch_size", 256)

        predicted_indices = []

        with torch.inference_mode():
            for start in range(0, len(features), batch_size):
                batch_features = torch.tensor(features[start:start + batch_size], dtype=torch.float32,device=device)

                logits = classifier(batch_features)

                predicted_indices.extend(logits.argmax(dim=1).cpu().tolist())

        return [state["index_to_label"][index] for index in predicted_indices]

class MultiClassConfidenceClassifierStrategy:
    """
    """

    def initialize(self, feature_dim: int, config: dict, device: torch.device) -> dict:
        return {
            "feature_dim": feature_dim,
            "device": device,
            "config": config,
            "classifier": None,
            "criterion": nn.CrossEntropyLoss(),
            "label_remapper": LabelRemapper(),
            "known_native_labels": set(),
            "index_to_label": {},
            "replay_by_label": {},
        }

    def train_stage(self, state: dict, stage: dict) -> tuple[dict, dict]:
        task_index = stage["task_index"]
        task_name = stage["task_name"]

        X_train = stage["X_train"]
        y_train = [str(label) for label in stage["y_train"]]

        config = state["config"]
        device = state["device"]

        current_labels = set(y_train)

        new_labels = sorted(current_labels - state["known_native_labels"])

        print("Current labels:", sorted(current_labels))
        print("Unseen labels:", new_labels)

        replay_features, _ = replay_to_arrays(
            replay_by_label=state["replay_by_label"],
            feature_dim=state["feature_dim"],
        )

        if task_index == 0:
            detected_new_cluster = False
        else:
            detected_new_cluster = (
                detect_current_dominated_cluster(
                    reference_features=replay_features,
                    current_features=X_train,
                    min_cluster_size=config["hdbscan_min_cluster_size"],
                    current_fraction_threshold=config["hdbscan_current_fraction"],
                )
            )

        print("HDBSCAN novelty signal:", detected_new_cluster)

        classifier = state["classifier"]
        label_remapper = state["label_remapper"]

        if task_index == 0:
            if len(new_labels) < 2:
                raise RuntimeError("Task 0 must contain at least two classes.")

            for label in new_labels:
                label_remapper.register(label)
                label_index = label_remapper.convert([label])[0]
                state["index_to_label"][label_index] = label

            state["known_native_labels"].update(new_labels)

            base_classifier = ConfidenceClassifier(embedding_dim=state["feature_dim"], output_dim=len(new_labels), dropout_prob=config.get("dropout_p", 0.1))
            classifier = base_classifier.to(device=device)

            trainable_parameters = list(
                classifier.parameters()
            )

            print(f"Created classifier with {len(new_labels)} outputs.")

        else:
            use_override = config.get("use_intervention_override",True)

            should_expand = (
                len(new_labels) > 0
                if use_override
                else detected_new_cluster
            )

            if new_labels and not should_expand:
                raise RuntimeError(
                    "New labels exist, but the novelty "
                    "detector did not request expansion."
                )

            if should_expand and new_labels:
                for label in new_labels:
                    label_remapper.register(label)

                    label_index = label_remapper.convert([label])[0]
                    state["index_to_label"][label_index] = label

                state["known_native_labels"].update(new_labels)

                trainable_parameters = list(
                    classifier.expand_classifier(num_to_add=len(new_labels))
                )

                print(f"Expanded classifier by {len(new_labels)} output(s): {new_labels}")

            else:
                trainable_parameters = [parameter for parameter in classifier.parameters() if
                                        parameter.requires_grad]

        classifier.eval()

        with torch.inference_mode():
            probe_count = min(2, len(X_train))

            probe = torch.tensor(X_train[:probe_count], dtype=torch.float32, device=device)

            actual_output_count, probabilities, isConfident = classifier(probe)
            actual_output_count = actual_output_count.shape[1]

        expected_output_count = len(state["known_native_labels"])


        print("Actual classifier outputs:", actual_output_count, )

        train_features, train_native_labels = (
            create_balanced_training_set(
                current_features=X_train,
                current_labels=y_train,
                replay_by_label=state["replay_by_label"],
                samples_per_class=config[
                    "balanced_samples_per_class"
                ],
                random_seed=config["random_seed"],
            )
        )

        print("Balanced classifier-training counts:")

        for label in sorted(set(train_native_labels)):
            print(f"  {label}: ", f"{train_native_labels.count(label)}")

        train_targets = label_remapper.convert(train_native_labels)

        optimizer = torch.optim.Adam(trainable_parameters, lr=config["learning_rate"], )

        classifier.train()

        train_targets = [int(min(x,1.0)) for x in train_targets]
        confidence_classifier_training_loop(classifier, state["criterion"], optimizer, train_features, train_targets,bot_label=1, device=device)
        # train_classifier(
        #     classifier,
        #     state["criterion"],
        #     optimizer,
        #     train_features,
        #     train_targets,
        #     epochs=config["epochs"],
        #     device=device,
        # )

        update_replay(
            replay_by_label=state["replay_by_label"],
            features=X_train,
            labels=y_train,
            replay_per_class=config["replay_per_class"],
            random_seed=config["random_seed"],
        )

        state["classifier"] = classifier

        train_info = {
            "task_name": task_name,
            "new_labels": new_labels,
            "hdbscan_detected_new_cluster": (detected_new_cluster),
            "total_classes": expected_output_count,
        }

        return state, train_info

    def predict_labels(self, state: dict, features: np.ndarray, evaluation_context: dict, ) -> list[str]:
        classifier = state["classifier"]
        device = state["device"]

        classifier.eval()

        batch_size = state["config"].get("eval_batch_size", 256)

        predicted_indices = []

        with torch.inference_mode():
            for start in range(0, len(features), batch_size):
                batch_features = torch.tensor(features[start:start + batch_size], dtype=torch.float32,device=device)

                logits, probabilities, confidence = classifier(batch_features)

                predicted_indices.extend(logits.argmax(dim=1).cpu().tolist())

        return [state["index_to_label"][index] for index in predicted_indices]

class MultiSVMClassifierStrategy:
    """
    """

    def initialize(self, feature_dim: int, config: dict, device: torch.device) -> dict:
        return {
            "feature_dim": feature_dim,
            "device": device,
            "config": config,
            "classifier": None,
            "criterion": nn.CrossEntropyLoss(),
            "label_remapper": LabelRemapper(),
            "known_native_labels": set(),
            "index_to_label": {},
            "replay_by_label": {},
            "replay_buffer": ReplayBuffer(50,0)
        }

    def train_stage(self, state: dict, stage: dict) -> tuple[dict, dict]:
        task_index = stage["task_index"]
        task_name = stage["task_name"]

        X_train = stage["X_train"]
        y_train = [str(label) for label in stage["y_train"]]

        config = state["config"]
        device = state["device"]

        current_labels = set(y_train)

        new_labels = sorted(current_labels - state["known_native_labels"])

        print("Current labels:", sorted(current_labels))
        print("Unseen labels:", new_labels)

        replay_features, _ = replay_to_arrays(
            replay_by_label=state["replay_by_label"],
            feature_dim=state["feature_dim"],
        )

        if task_index == 0:
            detected_new_cluster = False
        else:
            detected_new_cluster = (
                detect_current_dominated_cluster(
                    reference_features=replay_features,
                    current_features=X_train,
                    min_cluster_size=config["hdbscan_min_cluster_size"],
                    current_fraction_threshold=config["hdbscan_current_fraction"],
                )
            )

        print("HDBSCAN novelty signal:", detected_new_cluster)

        classifier = state["classifier"]
        label_remapper = state["label_remapper"]

        if task_index == 0:
            if len(new_labels) < 2:
                raise RuntimeError("Task 0 must contain at least two classes.")

            for label in new_labels:
                label_remapper.register(label)
                label_index = label_remapper.convert([label])[0]
                state["index_to_label"][label_index] = label

            state["known_native_labels"].update(new_labels)

            base_classifier = OneAgainstRestSVM(in_features=state["feature_dim"], output_dim=len(new_labels))
            classifier = base_classifier.to(device=device)

            trainable_parameters = list(
                classifier.parameters()
            )

            print(f"Created classifier with {len(new_labels)} outputs.")

        else:
            use_override = config.get("use_intervention_override", True)

            should_expand = (
                len(new_labels) > 0
                if use_override
                else detected_new_cluster
            )

            if new_labels and not should_expand:
                raise RuntimeError(
                    "New labels exist, but the novelty "
                    "detector did not request expansion."
                )

            if should_expand and new_labels:
                for label in new_labels:
                    label_remapper.register(label)

                    label_index = label_remapper.convert([label])[0]
                    state["index_to_label"][label_index] = label

                state["known_native_labels"].update(new_labels)

                trainable_parameters = list(
                    classifier.expand_classifier(num_to_add=len(new_labels))
                )

                print(f"Expanded classifier by {len(new_labels)} output(s): {new_labels}")

            else:
                trainable_parameters = [parameter for parameter in classifier.parameters() if
                                        parameter.requires_grad]

        classifier.eval()

        with torch.inference_mode():
            probe_count = min(2, len(X_train))

            probe = torch.tensor(X_train[:probe_count], dtype=torch.float32, device=device)

            actual_output_count = classifier(probe).shape[1]

        expected_output_count = len(state["known_native_labels"])

        print("Actual classifier outputs:", actual_output_count, )

        train_features, train_native_labels = (
            create_balanced_training_set(
                current_features=X_train,
                current_labels=y_train,
                replay_by_label=state["replay_by_label"],
                samples_per_class=config[
                    "balanced_samples_per_class"
                ],
                random_seed=config["random_seed"],
            )
        )

        print("Balanced classifier-training counts:")

        for label in sorted(set(train_native_labels)):
            print(f"  {label}: ", f"{train_native_labels.count(label)}")

        train_targets = label_remapper.convert(train_native_labels)



        classifier.train()

        multi_svm_training_loop(classifier, state["replay_buffer"], train_features, train_targets, epochs=config["epochs"], device=device)


        update_replay(
            replay_by_label=state["replay_by_label"],
            features=X_train,
            labels=y_train,
            replay_per_class=config["replay_per_class"],
            random_seed=config["random_seed"],
        )

        state["classifier"] = classifier

        train_info = {
            "task_name": task_name,
            "new_labels": new_labels,
            "hdbscan_detected_new_cluster": (detected_new_cluster),
            "total_classes": expected_output_count,
        }

        return state, train_info

    def predict_labels(self, state: dict, features: np.ndarray, evaluation_context: dict, ) -> list[str]:
        classifier = state["classifier"]
        device = state["device"]

        classifier.eval()

        batch_size = state["config"].get("eval_batch_size", 256)

        predicted_indices = []

        with torch.inference_mode():
            for start in range(0, len(features), batch_size):
                batch_features = torch.tensor(features[start:start + batch_size], dtype=torch.float32,
                                              device=device)

                logits = classifier(batch_features)

                predicted_indices.extend(logits.argmax(dim=1).cpu().tolist())

        return [state["index_to_label"][index] for index in predicted_indices]