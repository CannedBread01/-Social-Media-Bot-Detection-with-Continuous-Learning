import copy
import math
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def create_profile_vector(user_data):
    """
    creates the meta-data embedding vector for the provided user.
    :param user_data: a dictionary corresponding to one sample from the dataset. Should contain a 'profile' entry.
    :return: tensor of the meta-data embedding vector
    """
    # user profile embedding
    profile_embedding = torch.tensor([
        len("" if user_data.name is None else user_data.name),
        len("" if user_data.screen_name is None else user_data.screen_name),
        math.log(user_data.statuses_count+1),
        math.log(user_data.followers_count+1),
        math.log(user_data.friends_count+1),
        math.log(user_data.favourites_count+1),
        user_data.verified,
    ])
    return profile_embedding

def create_tweet_vectors(tweet_data, tokenizer, model, batch_size = math.inf, max_tweets = math.inf, device ='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    creates the tweets embedding vector for the provided user.
    :param device: the device on which to process the data.
    :param model: the model used for creating semantic embeddings from tweets
    :param tokenizer: the tokenizer used for creating tokens from text strings
    :param tweet_data: a dictionary corresponding to one sample from the dataset. Should contain a 'tweets' entry.
    :param batch_size: the batch size used to process the tweets. Defaults to processing all tweets at once
    :param max_tweets: the maximum amount of tweets to process per profile. Defaults to processing all available tweets at once.
    :return: tensor of the tweets embedding vector
    """
    tweet_count = len(tweet_data)
    # if no tweets are available, return a default vector
    if tweet_count == 0:
        return torch.atleast_2d(torch.zeros(768)).detach().cpu()

    # get the maximum wanted tweets to process
    texts = list(map(lambda x: x.text, tweet_data))[0: min(tweet_count, max_tweets)]
    # process texts in batches
    raw_embeddings = []
    index = 0
    while index < min(tweet_count, max_tweets):
        # grab text batch
        text_batch = texts[index:min(index+batch_size, tweet_count)]
        # replace urls with 'url'
        URL_PATTERN = r"https?://\S+|www\.\S+"
        text_batch = [re.sub(URL_PATTERN, " url ", text) for text in text_batch]

        # get textual embedding of the batch
        with torch.no_grad():
            tokenized_text = tokenizer(text_batch, padding="longest", truncation=True, return_tensors="pt").to(device)
            output = model(**tokenized_text)
        # collect embeddings
        raw_embeddings.extend(torch.unbind(output.last_hidden_state[:, 0, :].detach().cpu(),dim=0))
        final_embeddings = torch.stack(raw_embeddings, dim=0)

        index += batch_size

    return torch.atleast_2d(final_embeddings).detach().cpu()


class EarlyStopper:
    """
    Tracks the progress the model makes during training and terminates early if no progress is made.
    Additionally the current best performing model weights are stored, so that they can be retrieved.
    """
    def __init__(self, patience: int = 3):
        """
        :param patience: determines how long the Stopper should tolerate worsening performance before triggering
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_weights = None
        self.early_stop = False

    def evaluate_step(self, current_loss, model):
        """
        Performs one step in the evaluation of the model
        :param current_loss: the current loss value associated with the passed model
        :param model: the model that is being evalauted
        """
        if current_loss < self.best_loss: # reset patience and store best model
            self.best_loss = current_loss
            self.counter = 0
            self.best_model_weights = copy.deepcopy(model.state_dict())
        else: # increase patience counter
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


    def load_best_model(self, model):
        """
        Loads the best model weights achieved during training into the provided model
        :param model: the model that was being evalauted
        :return: returns the model with the best weights achived during training. model remains unchanged if no weights were stored yet
        """

        if self.best_model_weights is not None:
            model.load_state_dict(self.best_model_weights)
            print(f"Loaded best model weights (Best Loss: {self.best_loss:.4f})")
        else:
            print("No weights have been saved yet!")
        return model

def train_classifier(classifier, criterion, optimizer ,input_samples, ground_truth_labels, epochs = 15, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    classifier.train()
    # create dataset from ground_truths and embedding features for batch processing
    input_tensors = torch.tensor(np.array(input_samples))
    ground_truth_tensors = torch.tensor(ground_truth_labels)
    dataset = TensorDataset(input_tensors, ground_truth_tensors)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # setup early stopper
    early_stopper = EarlyStopper(patience=3)
    # training loop
    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0

        predictions = []
        ground_truths = []

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.float().to(device)
            targets = targets.to(device)

            # pass data through classifier and calculate loss
            outputs = classifier(inputs)
            loss = criterion(outputs, targets)
            # optimize classifier
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # prediction
            prediction = torch.argmax(outputs, dim=1)

            # statistics
            predictions.extend(prediction.detach().cpu().numpy())
            ground_truths.extend(targets.detach().cpu().numpy())
            running_loss += loss.item() * inputs.size(0)

        # Calculate average loss across the entire epoch
        epoch_loss = running_loss / len(input_samples)
        # check performance with early stopper
        early_stopper.evaluate_step(epoch_loss, classifier)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f}")
        if early_stopper.early_stop:
            print("-- stopping early --")
            break
    # get best model weights from training
    classifier = early_stopper.load_best_model(classifier)
    return ground_truths, predictions

def test_classifier(classifier, input_samples, ground_truth_labels, device = 'cuda' if torch.cuda.is_available() else 'cpu'):

    # create dataset from ground_truths and embedding features for batch processing
    input_tensors = torch.tensor(np.array(input_samples))
    ground_truth_tensors = torch.tensor(ground_truth_labels)
    dataset = TensorDataset(input_tensors, ground_truth_tensors)
    test_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    classifier.eval()

    predictions = []
    ground_truths = []

    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        inputs = inputs.float().to(device)
        targets = targets.to(device)

        # pass data through classifier and calculate loss
        with torch.inference_mode():
            outputs = classifier(inputs)

        # prediction
        prediction = torch.argmax(outputs, dim=1)

        predictions.extend(prediction.detach().cpu().numpy())
        ground_truths.extend(targets.detach().cpu().numpy())
    return ground_truths, predictions


def multi_svm_training_loop(classifier, replay_buffer, input_samples, ground_truth_labels, epochs = 15, batch_size = 128, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    # general setup
    C = 1.0
    learning_rate = 0.1
    optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)

    classifier.train()
    # create dataset from ground_truths and embedding features for batch processing
    input_tensors = torch.tensor(np.array(input_samples))
    ground_truth_tensors = torch.tensor(ground_truth_labels)
    dataset = TensorDataset(input_tensors, ground_truth_tensors)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # setup early stopper
    early_stopper = EarlyStopper(patience=3)
    # training loop
    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0

        predictions = []
        ground_truths = []

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            dataset_inputs = inputs.float().to(device)
            dataset_targets = targets.to(device)

            # sample buffer and concatenate it to dataset sample
            buffer_targets, buffer_inputs = replay_buffer.get_samples(round(batch_size/8))

            inputs = torch.concat((dataset_inputs, buffer_inputs.to(device)))
            targets = torch.concat((dataset_targets, buffer_targets.to(device))).int()

            # encode targets in a binary fashion
            remapped_targets = torch.full((len(targets), classifier.size), -1).to(device)
            remapped_targets[torch.arange(len(targets)), targets] = 1

            # pass data through classifier and calculate loss
            optimizer.zero_grad()
            outputs = classifier(inputs)


            # calculate loss
            regularization_loss = 0.5 * torch.sum(classifier.get_weights() ** 2)
            hinge_loss = torch.mean(torch.clamp(1 - remapped_targets * outputs, min=0))
            loss = regularization_loss + C * hinge_loss
            loss.backward()
            optimizer.step()

            # prediction
            prediction = torch.argmax(outputs, dim=1)
            # add samples to the buffer
            replay_buffer.add_samples(targets[:16].detach(), inputs[:16].detach())

            # statistics
            predictions.extend(prediction[:batch_size].detach().cpu().numpy())
            ground_truths.extend(targets[:batch_size].detach().cpu().numpy())
            running_loss += loss.item() * inputs.size(0)

        # Calculate average loss across the entire epoch
        epoch_loss = running_loss / len(input_samples)
        # check performance with early stopper
        early_stopper.evaluate_step(epoch_loss, classifier)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f}")
        if early_stopper.early_stop:
            print("-- stopping early --")
            break
    # get best model weights from training
    classifier = early_stopper.load_best_model(classifier)
    return ground_truths, predictions

def confidence_classifier_training_loop(classifier, criterion, optimizer ,input_samples, ground_truth_labels, epochs = 15, bot_label = 0, batch_size = 128, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    classifier.train()
    # create dataset from ground_truths and embedding features for batch processing
    input_tensors = torch.tensor(np.array(input_samples))
    ground_truth_tensors = torch.tensor(ground_truth_labels)
    dataset = TensorDataset(input_tensors, ground_truth_tensors)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    inconfident_samples_buffer = []

    # setup early stopper
    early_stopper = EarlyStopper(patience=3)
    # training loop
    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0

        predictions = []
        ground_truths = []

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.float().to(device)
            targets = targets.to(device)

            # sample from buffer
            sample_size = min(len(inconfident_samples_buffer), 16)
            buffer_sample = inconfident_samples_buffer[:sample_size]
            inconfident_samples_buffer = inconfident_samples_buffer[sample_size:]

            if sample_size != 0:
                buffer_sample = torch.stack(buffer_sample).to(device)
                buffer_targets = torch.full((sample_size, ), bot_label).to(device)

                # concat
                inputs = torch.concat((inputs, buffer_sample)).to(device)
                targets = torch.concat((targets, buffer_targets)).to(device)

            # pass data through classifier and calculate loss
            outputs, prob, is_confident = classifier(inputs)
            loss = criterion(outputs, targets)

            # optimize classifier
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # prediction
            prediction = torch.argmax(outputs, dim=1)

            # add to buffer if confidence is below the threshold
            to_add = inputs[torch.sum(is_confident,1) == 0].detach()
            inconfident_samples_buffer.extend(to_add)

            # statistics
            predictions.extend(prediction.detach().cpu().numpy())
            ground_truths.extend(targets.detach().cpu().numpy())
            running_loss += loss.item() * inputs.size(0)

        # Calculate average loss across the entire epoch
        epoch_loss = running_loss / len(input_samples)
        # check performance with early stopper
        early_stopper.evaluate_step(epoch_loss, classifier)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f}")
        if early_stopper.early_stop:
            print("-- stopping early --")
            break
    # get best model weights from training
    classifier = early_stopper.load_best_model(classifier)
    return ground_truths, predictions