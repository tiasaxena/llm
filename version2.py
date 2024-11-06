# Import all the necessary libraries
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 64  # How many independant sequences will be processed in parallel
block_size = 256 # What is the maximum context length for prediction
max_iter = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# Download the shakespeare dataset we want the model to train on
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read the text written
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Find the vocabulary size and all the unique characters in the string
unique_chars = sorted(list(set(text)))
vocab_size = len(unique_chars)
# Create mapping from string to integer and integer to string
stoi = {ch: i for i, ch in enumerate(unique_chars)}
itos = {i: ch for i, ch in enumerate(unique_chars)}

# Encode & Decode
encode = lambda str: [
    stoi[ch] for ch in str
]  # Takes the string, for every character returns the interger mapping, outputs a list of integers
decode = lambda int_list: "".join(
    [itos[idx] for idx in int_list]
)  # Takes in the list of integers, maps int to char at each index, outputs a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
len_train_data = int(0.9 * len(data))
train_data = data[:len_train_data]
val_data = data[len_train_data:]


# Data Loading
def get_batch(split):
    """
    Generate a small batch of data from inputs x and targets y
    """

    data = train_data if split == "train" else val_data
    # ix stores the random indexes from where the 4 string data chunks will be taken
    ix = torch.randint(
        len(data) - block_size, (batch_size,)
    )  # Maximum limit --> len(data) - block_size, dimension of o/p tensor --> 4
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    output = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss
        output[split] = losses.mean().item()
        model.train()
    return output


# Head Class
class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        # ! Dropouts mask certain nodes each time, which helps ni learning via ensemble methods.. It is a Regularization tehnique.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # Compute scaled attention score / affinities
        weights = (
            q @ k.transpose(-2, -1) / (C**0.5)
        )  # (B, T, C) @ (B, C, T) = (B, T, T)
        # Nullify the upper triangular part of the matrix
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # Compute the attention probabilities
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)
        # Apply the attention to the values
        v = self.value(x)  # (B, T, C)
        output = weights @ v  # (B, T, T) @ (B, T, C) = (B, T, C)
        return output


# Multi-Head Attention Class
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projections = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim = -1) # Along Channel dimension
        output = self.projections(output)
        return output
    

# FeedForward Layer Class
class FeedForward(nn.Module):
    """ A simple linear layer followed by non-linearity. """
    def __init__(self, n_embd):
        super().__init__()
        # It is done on per token level, that is, alll the tokens in the sequence do this indeppendantly of each other
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # 4 times the embedding size, acc to the attention is all you need paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Projection Layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Block Class
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention_head = MultiHeadAttention(num_heads=n_head, head_size=head_size)
        self.ffwd = FeedForward(n_embd)
        self.layerNorm1 = nn.LayerNorm(n_embd)
        self.layerNorm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        return x + self.ffwd(self.layerNorm2(x + self.self_attention_head(self.layerNorm1(x)))) # The x + indicates that we are using residual connections/skip connections

# Bigram Language Model Class
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # We dont wan to directly look up to the embedding table unlike in bigram.py, so we create a level of indirection
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=n_embd
        )  # n_embd * n_embd table (n_embd = # of embedding dimensions)
        self.position_embedding_table = nn.Embedding(
            num_embeddings=block_size, embedding_dim=n_embd
        )  # block_size * n_embd table
        # self.self_attention_head = Head(n_embd)
        # ! We will have MultiHeadAttention and FeedForward in the `Block` class
        # self.self_attention_head = MultiHeadAttention(num_heads=4, head_size=n_embd//4) # i.e. 4 heads of 8 dimensional self-attention
        # self.ffwd = FeedForward(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.layerNorm = nn.LayerNorm(n_embd) # Final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)  # lm_head = language model head

    def forward(self, input, targets=None):
        # input = xb and targets = yb
        B, T = input.shape

        token_emb = self.token_embedding_table(
            input
        )  # (Batch(=4) * Time(=8) * Channel(= embd_size))
        positional_embedding = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (Time(=8) * Channel(= embd_size))
        x = (
            token_emb + positional_embedding
        )  # (Batch(=4) * Time(=8) * Channel(= embd_size))
        # ! We will have MultiHeadAttention and FeedForward in the `Block` class
        # x = self.self_attention_head(x)  # (Batch(=4) * Time(=8) * Channel(= embd_size))
        # x = self.ffwd(x)  # (Batch(=4) * Time(=8) * Channel(= embd_size))/
        x = self.blocks(x)  # (Batch(=4) * Time(=8) * Channel(= embd_size))
        logits = self.lm_head(x)  # (Batch(=4) * Time(=8) * Channel(= vocab_size))

        # F.cross_entropy will expect Batch*Channel*Time, however we have B*T*C, so we will reshape
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, input, max_new_tokens):
        """
        This function is the part where we achieve character generation from the model
        """
        # input is B*T array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop index to the last block_size
            input = input[:, -block_size:]
            # Get the predictions
            logits, loss = self(input)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            input_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            input = torch.cat((input, input_next), dim=1)  # (B, T+1)
        return input

 
model = BigramLanguageModel()
m = model.to(device)
# Print the number of parameters in our model
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6}")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iter):
    # Every once in a while we will evaluate the model on the validation set
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(losses)
        print(
            f"iter {iter}, train_loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}"
        )

    # Sample a batch of data
    xb, yb = get_batch("train")

    # Evaluate the loss
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()

# Generate from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
