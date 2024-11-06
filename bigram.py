# Import all the necessary libraries
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32  # How many independant sequences will be processed in parallel
block_size = 8  # What is the maximum context length for prediction
max_iter = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

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


# Bigram Language Model Class
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Every single in the input will refer to the embedding table and will pluck out a row of the embdding table corresponding to its index
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=vocab_size
        )  # 65 * 65 table

    def forward(self, input, targets=None):
        # input = xb and targets = yb
        logits = self.token_embedding_table(
            input
        )  # (Batch(=4) * Time(=8) * Channel(=65, the vocab size))

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


model = BigramLanguageModel(vocab_size)
m = model.to(device)

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
