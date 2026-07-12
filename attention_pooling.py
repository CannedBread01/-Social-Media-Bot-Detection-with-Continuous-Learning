import torch
from torch import nn
import re

class AttentionPooling(nn.Module):

    def __init__(self,embed_dim=768, hidden_dim=128):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, tweet_embeds, mask=None):

        scores = self.attention(tweet_embeds).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * tweet_embeds, dim=1)

        return pooled, weights


class BotDetectionModel(nn.Module):
    def __init__(self, tweet_embed_dim=768, profile_dim=7, hidden_dim=128, num_classes=4):
        super().__init__()
        self.attention_pooling = AttentionPooling(tweet_embed_dim, hidden_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(tweet_embed_dim + profile_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, tweet_embeds, profile_vec, mask=None):
        pooled_tweets, attn_weights = self.attention_pooling(tweet_embeds, mask)
        combined = torch.cat([pooled_tweets, profile_vec], dim=1)
        logits = self.classifier(combined)
        return logits, attn_weights

def create_tweet_vectors_attention(tweet_data, tokenizer, model, max_tweets=50, device='cuda'):
    tweet_count = len(tweet_data)
    embed_dim = 768

    if tweet_count == 0:
        return (torch.zeros(max_tweets, embed_dim),
                torch.zeros(max_tweets, dtype=torch.bool))

    texts = [t.text for t in tweet_data[:max_tweets]]
    URL_PATTERN = r"https?://\S+|www\.\S+"
    texts = [re.sub(URL_PATTERN, " url ", t) for t in texts]

    with torch.no_grad():
        tokenized = tokenizer(texts, padding="longest", truncation=True, return_tensors="pt").to(device)
        output = model(**tokenized)
    embeds = output.last_hidden_state[:, 0, :].detach().cpu()

    n = embeds.shape[0]
    padded = torch.zeros(max_tweets, embed_dim)
    mask = torch.zeros(max_tweets, dtype=torch.bool)
    padded[:n] = embeds
    mask[:n] = True

    return padded, mask