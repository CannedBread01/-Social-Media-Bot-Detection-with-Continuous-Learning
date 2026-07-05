from collections.abc import Callable, Iterable
import math
import re

import numpy as np
import torch
import umap
from sklearn.preprocessing import RobustScaler
from transformers import AutoModel, AutoTokenizer


URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)


def safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def safe_log1p(value) -> float:
    return math.log1p(max(safe_float(value), 0.0))


def create_profile_vector(user) -> torch.Tensor:
    """
    Creates the 8 metadata features used in the old notebooks.
    """

    values = [
        len(str(user.name or "")),
        len(str(user.screen_name or "")),
        safe_log1p(user.statuses_count),
        safe_log1p(user.followers_count),
        safe_log1p(user.friends_count),
        safe_log1p(user.favourites_count),
        0.0,  # protected is unavailable in the common UserData dataclass
        float(bool(user.verified)),
    ]

    return torch.tensor(values, dtype=torch.float32)


def mean_pool(last_hidden_state, attention_mask):
    """
    Mean-pool DistilBERT token embeddings while ignoring padding tokens.
    """

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)

    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)

    return summed / counts


class FeaturePipeline:
    """
    Shared stage before any classifier.

    Dataset Sample
        -> profile metadata + frozen DistilBERT tweet embedding
        -> RobustScaler
        -> UMAP
        -> classifier-ready feature vectors
    """

    def __init__(self, config: dict, device: torch.device):
        self.device = device

        self.max_tweets_per_user = config.get(
            "max_tweets_per_user",
            20,
        )

        self.tweet_batch_size = config.get(
            "tweet_batch_size",
            32,
        )

        self.max_token_length = config.get(
            "max_token_length",
            128,
        )

        self.random_seed = config.get(
            "random_seed",
            42,
        )

        model_name = config.get(
            "embedding_model",
            "distilbert-base-uncased",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.bert_model = AutoModel.from_pretrained(
            model_name
        ).to(self.device)

        self.bert_model.eval()

        for parameter in self.bert_model.parameters():
            parameter.requires_grad_(False)

        self.scaler = RobustScaler(
            with_centering=False
        )

        self.dim_reducer = umap.UMAP(
            n_components=config.get(
                "umap_components",
                15,
            ),
            n_neighbors=config.get(
                "umap_neighbors",
                20,
            ),
            min_dist=config.get(
                "umap_min_dist",
                0.1,
            ),
            metric=config.get(
                "umap_metric",
                "cosine",
            ),
            random_state=self.random_seed,
        )

        self.is_fitted = False

    def embed_split(
        self,
        dataset: Iterable,
        task_name: str,
        label_transform: Callable[[str], str],
        max_samples: int | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Converts a dataset split into raw feature vectors and string labels.

        Does NOT scale, reduce, train, replay, or evaluate.
        """

        features = []
        labels = []

        print(f"Embedding '{task_name}'...")

        with torch.inference_mode():
            for sample_index, sample in enumerate(dataset):
                if (
                    max_samples is not None
                    and sample_index >= max_samples
                ):
                    break

                if sample_index % 100 == 0:
                    print(f"  {sample_index} users")

                profile_vector = create_profile_vector(
                    sample.user_data
                )

                texts = []

                for tweet in sample.tweet_data[
                    :self.max_tweets_per_user
                ]:
                    text = str(tweet.text or "").strip()

                    if text:
                        texts.append(
                            URL_PATTERN.sub(
                                " url ",
                                text,
                            )
                        )

                if not texts:
                    user_tweet_vector = torch.zeros(
                        self.bert_model.config.hidden_size,
                        dtype=torch.float32,
                    )

                else:
                    pooled_batches = []

                    for start in range(
                        0,
                        len(texts),
                        self.tweet_batch_size,
                    ):
                        batch_texts = texts[
                            start:start + self.tweet_batch_size
                        ]

                        tokens = self.tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=self.max_token_length,
                            return_tensors="pt",
                        )

                        tokens = {
                            name: tensor.to(self.device)
                            for name, tensor in tokens.items()
                        }

                        outputs = self.bert_model(**tokens)

                        pooled_batches.append(
                            mean_pool(
                                outputs.last_hidden_state,
                                tokens["attention_mask"],
                            ).cpu()
                        )

                    user_tweet_vector = torch.cat(
                        pooled_batches,
                        dim=0,
                    ).mean(dim=0)

                combined = torch.cat(
                    [
                        profile_vector,
                        user_tweet_vector.float(),
                    ],
                    dim=0,
                )

                features.append(
                    combined.numpy().astype(np.float32)
                )

                labels.append(
                    label_transform(str(sample.label))
                )

        if not features:
            feature_dim = (
                8 + self.bert_model.config.hidden_size
            )

            return (
                np.empty(
                    (0, feature_dim),
                    dtype=np.float32,
                ),
                [],
            )

        result = np.stack(features).astype(np.float32)

        print(
            f"Finished '{task_name}': "
            f"{result.shape}, "
            f"labels={sorted(set(labels))}"
        )

        return result, labels

    def fit_transform_first_task(
        self,
        raw_features: np.ndarray,
    ) -> np.ndarray:
        """
        Fit RobustScaler and UMAP on Task 0 only.
        """

        if len(raw_features) == 0:
            raise ValueError(
                "Cannot fit feature pipeline on an empty task."
            )

        scaled_features = self.scaler.fit_transform(
            raw_features
        )

        reduced_features = self.dim_reducer.fit_transform(
            scaled_features
        )

        self.is_fitted = True

        return np.asarray(
            reduced_features,
            dtype=np.float32,
        )

    def transform(
        self,
        raw_features: np.ndarray,
    ) -> np.ndarray:
        """
        Transform all later tasks using Task 0's fitted scaler and UMAP.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "FeaturePipeline must be fitted on Task 0 "
                "before later tasks are transformed."
            )

        scaled_features = self.scaler.transform(
            raw_features
        )

        reduced_features = self.dim_reducer.transform(
            scaled_features
        )

        return np.asarray(
            reduced_features,
            dtype=np.float32,
        )