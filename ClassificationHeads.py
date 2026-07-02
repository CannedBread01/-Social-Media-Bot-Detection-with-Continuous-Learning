import torch
from torch import nn


class BaselineClassificationHead(nn.Module):
    """
    The baseline classification head is a normal neural network without any continuous learning aspects.
    It is used to determine the upper bound of our backbone.
    """
    def __init__(self, embedding_dim, output_dim, dropout_prob = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden = nn.Linear(embedding_dim, max(round(embedding_dim/2), output_dim))
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(in_features=max(round(embedding_dim/2), output_dim), out_features=output_dim)
    def forward(self, x):
        x = self.dropout(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x

    def expand_classifier(self, num_to_add=1):
        """
        does nothing,as it functions as a dummy, in order to keep the interface consistent.
        """
        return



class ProgressiveNeuralNetworkClassifier(nn.Module):
    """
    A simple progressive neural network that expands in size to accommodate new classes.
    On expansion a parallel classification head is added to accommodate new classes.
    The old classification layers are removed from further optimization once a new layer is added
    """
    def __init__(self, in_features, output_dim, dropout_p = 0.1):
        super().__init__()
        self.in_features = in_features
        self.num_classes = output_dim
        self.device = None # needed so that new layers are on the correct device

        self.dropout = nn.Dropout(p=dropout_p)
        # contains all output layers for classification
        self.classificationLayers = nn.ModuleList([
            nn.Linear(in_features=in_features, out_features=output_dim)
        ])

    def forward(self, x):
        # set the device once
        if self.device is None: self.device = x.device

        x = self.dropout(x)
        # pass through all classification layers in parallel
        outputs = list(map(lambda layer: layer(x), self.classificationLayers))
        outputs = torch.cat(outputs, dim=-1) # concat outputs
        return outputs

    def expand_classifier(self, num_to_add = 1):
        """
        dynamically expands the classifier by the number of classes specified.
        Weights for the old classes are removed from the computation graph that is responsible for optimization
         and can't be trained anymore
        :param num_to_add: the number of classes to add
        :return: the new relevant parameters of the model to optimize
        """
        # disable optimization for old layers
        for layer in self.classificationLayers:
            for param in layer.parameters():
                param.requires_grad = False
        # add new layer to list
        new_layer = nn.Linear(self.in_features, num_to_add).to(device=self.device)
        self.classificationLayers.append(new_layer)
        return new_layer.parameters()



class SVM(nn.Module):
    """
    a single support vector machine (SVM) implemented as a single linear layer.
    Use hinge loss and L2 regularization for it to behave like a SVM.
    """
    def __init__(self, in_features):
        """
        :param in_features: dimension of the input features
        """
        super().__init__()
        self.classifier = nn.Linear(in_features=in_features, out_features=1)
    def forward(self, x):
        """
        :param x: input tensor
        :return: tensor of logits, tensor of predictions
        """
        x = self.classifier(x)
        return x, torch.sign(x)

    def get_weights(self):
        """
        returns the weights of this model.
        :return: weights of this model
        """
        return self.classifier.weight

class OneAgainstRestSVM(nn.Module):
    """

    """
    def __init__(self, in_features, output_dim):
        super().__init__()
        self.svms = nn.ModuleList([SVM(in_features=in_features) for _ in list(range(output_dim))])
        self.in_features_dim = in_features
        self.size = output_dim
        self.device = None  # needed so that new layers are on the correct device

    def forward(self, x):
        # set the device once
        if self.device is None: self.device = x.device
        # handle forward pass
        outputs = list(map(lambda svm: svm(x)[0], self.svms))
        outputs = torch.cat(outputs, dim=-1) # concat outputs
        return outputs

    def get_weights(self):
        """
        returns the weights of this model.
        :return: weights of this model
        """
        return torch.stack([ svm.classifier.weight for svm in self.svms ])
    def expand_classifier(self, num_to_add = 1):
        """
        dynamically expands the classifier by the number of classes specified.
        :param num_to_add: the number of classes to add
        :return: the new relevant parameters of the model to optimize
        """

        for i in range(num_to_add):
            new_svm = SVM(in_features=self.in_features_dim).to(device=self.device)
            self.svms.append(new_svm)
            self.size += 1
        return self.parameters()


class ReplayBuffer:
    """
    Replay Buffer that manages samples per class.
    If added samples would exceed the buffer limit, then random old samples are replaced instead
    """

    def __init__(self, buffer_size: int, seed: int):
        """
        :param buffer_size: the maximum amount of samples to store per class/label
        :param seed: seed, to keep sampling reproducible
        """
        self.buffer_size = buffer_size

        # PyTorch random generator for reproducibility
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        # dictionary to act as a buffer for each class/label
        self.buffer = {}

    def add_samples(self, labels, samples):
        """
        adds samples to the replay buffer.
        If the buffer is full, the provided samples are used to replace existing samples.
        :param labels: list of labels for the samples
        :param samples: list of samples to store
        """

        # check if input dimensions match
        assert (len(labels) == len(samples)), "Labels and samples must have the same length."
        labels = labels.detach()
        samples = samples.detach()

        for label, sample in zip(labels, samples):
            label = label.item()
            # Initialize list if class hasn't been seen yet
            if label not in self.buffer:
                self.buffer[label] = []

            # add sample to buffer, if buffer size is reached use it to replace an old sample
            class_buffer = self.buffer[label]
            if len(class_buffer) < self.buffer_size: # just append
                class_buffer.append(sample)
            else: # randomly replace one
                index = torch.randint(0, self.buffer_size, (1,), generator=self.rng).item()
                class_buffer[index] = sample

    def get_samples(self, samples_per_class: int):
        """
        Provides a random sample from each stored sample. Samples are without duplicates,
        meaning if fewer samples are available in the buffer than requested, fewer samples will be returned
        :param samples_per_class: how many samples per class to return
        :return: a tensor of labels, and a tensor of samples
        """

        out_labels = []
        out_samples = []

        # loop through each stored class buffer and add to the sample
        for key, class_buffer in self.buffer.items():
            current_count = len(class_buffer)
            # skip, if no samples are available
            if current_count == 0: continue
            # limit samples to available amount
            num_to_draw = min(samples_per_class, current_count)

            # Generate random indices without replacement
            indices = torch.randperm(current_count, generator=self.rng)[:num_to_draw]

            # go through random indexes and add samples to list
            for idx in indices:
                out_labels.append(key)
                out_samples.append(class_buffer[idx.item()])

        if out_samples == []: return torch.tensor([]), torch.tensor([])
        sampled_labels = torch.tensor(out_labels)
        sampled_data = torch.stack(out_samples)

        return sampled_labels, sampled_data

class ConfidenceClassifier(nn.Module):
    def __init__(self, embedding_dim, output_dim, confidence_threshold = 0.7, dropout_prob = 0.1):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden = nn.Linear(embedding_dim, max(round(embedding_dim/2), output_dim))
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(in_features=max(round(embedding_dim/2), output_dim), out_features=output_dim)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.dropout(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x, self.softmax(x), self.softmax(x) > self.confidence_threshold # logits and probabilities

    def expand_classifier(self, num_to_add=1):
        """
        does nothing,as it functions as a dummy, in order to keep the interface consistent.
        """
        return