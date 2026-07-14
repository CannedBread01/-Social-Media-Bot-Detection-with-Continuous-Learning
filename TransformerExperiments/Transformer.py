import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0] #number of examples in the batch
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(queries)  # (N, query_len, heads, head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)


    
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N,
            query_len,
            self.heads * self.head_dim
        )  # after einsum (N,query_len, head, head_dim) then flatten  the last two dimensions (N, query_len, embed_size)

        out = self.fc_out(out)  # (N, query_len, embed_size)
        
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, activation_function = nn.ReLU()):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            activation_function,
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        activation_function = nn.ReLU()
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    activation_function = nn.ReLU(),
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
        
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion,device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
    
class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, dropout, forward_expansion,device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out
    
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        # (N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

# modified encoder
class TransformerBlockModified(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, activation_function=nn.ReLU()):
        super(TransformerBlockModified, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm0 = nn.LayerNorm(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            activation_function,
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        value = self.norm0(value)
        key = self.norm0(key)
        query = self.norm0(query)

        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class EncoderModified(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
            activation_function=nn.ReLU()
    ):
        super(EncoderModified, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlockModified(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    activation_function=nn.ReLU(),
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


# ---- Not relevant ----

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define toy vocabularies (including special tokens)
    # 0 = <pad>, 1 = <sos> (Start of Sentence), 2 = <eos> (End of Sentence)
    src_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "The": 3, "weather": 4, "in": 5, "spring": 6, "is": 7, "nice": 8}
    trg_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "Das": 3, "Wetter": 4, "im": 5, "Frühling": 6, "ist": 7, "schön": 8}
    
    # Reverse vocabulary for decoding the output
    inv_trg_vocab = {v: k for k, v in trg_vocab.items()}

    # Tokenize and Numericalize the sentences
    src_sentence = ["<sos>", "The", "weather", "in", "spring", "is", "nice", "<eos>"]
    trg_sentence = ["<sos>", "Das", "Wetter", "im", "Frühling", "ist", "schön", "<eos>"]

    src_indices = [src_vocab[token] for token in src_sentence]
    trg_indices = [trg_vocab[token] for token in trg_sentence]

    # Convert to Tensors and add a Batch dimension (B=1)
    src_tensor = torch.tensor([src_indices]).to(device)  # Shape: (1, src_len)
    trg_tensor = torch.tensor([trg_indices]).to(device)  # Shape: (1, trg_len)

    #  Initialize the model matching our toy vocabulary sizes
    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)
    src_pad_idx = 0
    trg_pad_idx = 0

    model = Transformer(
        src_vocab_size=src_vocab_size, 
        trg_vocab_size=trg_vocab_size, 
        src_pad_idx=src_pad_idx, 
        trg_pad_idx=trg_pad_idx
    ).to(device)
    
    # Put model in evaluation mode
    model.eval()

    # Run Autoregressive Inference (How translation works at test-time)
    # We start with just the <sos> token in our target sequence
    outputs = [trg_vocab["<sos>"]]
    max_length = 30

    print("\nTranslating...")
    with torch.no_grad():
        for _ in range(max_length):
            trg_input = torch.tensor([outputs]).to(device) # Shape: (1, current_trg_len)
            
            # Forward pass through the Transformer
            out = model(src_tensor, trg_input) # Shape: (1, current_trg_len, trg_vocab_size)
            
            # Get the predicted token for the absolute last position
            best_next_item = out.argmax(dim=2)[:, -1].item()
            
            outputs.append(best_next_item)
            
            # Stop if the model outputs the End of Sentence token
            if best_next_item == trg_vocab["<eos>"]:
                break

    # Decode the predicted token IDs back into words
    translated_sentence = [inv_trg_vocab[idx] for idx in outputs]
    
    print("\n--- RESULTS ---")
    print(f"Source English: {' '.join(src_sentence[1:-1])}")
    print(f"Expected German: {' '.join(trg_sentence[1:-1])}")
    print(f"Model Raw Output Tokens: {translated_sentence}")
