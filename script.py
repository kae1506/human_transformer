import numpy as np
import torch
import torch.nn as nn

## DATA ORGANISATION ## 

# get text
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# sort text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# making lookup tables

# string to index
stoi = { ch:i for i, ch in enumerate(chars) }
# index to string
itos = { i:ch for i, ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda e: ''.join([itos[c] for c in e])

# batching, train and val split
full_encoding = torch.tensor(encode(text))
s = int(len(full_encoding)*0.9)  # 90% will be training
train_data = full_encoding[:s]
val_data = full_encoding[s:]

T = 8   # no. of characters per input
B = 10    # no. of inputs, per batch

def get_batch(split):
  data = train_data if split == 'train' else val_data

  # first, generate a list of indices
  indices = torch.randint(len(full_encoding) - T, (B, ))
  # then, generate inputs from each index
  print(indices.shape)
  inputs = [full_encoding[index: index+T] for index in indices]
  targets = [full_encoding[index+1: index+1+T] for index in indices]

  # formulate into tensors
  inputs = torch.stack(inputs)
  targets = torch.stack(targets)

  return inputs, targets


head_size = 16
batch_size = 32
time = 8
B, T, C = batch_size, time, vocab_size


## CREATING TRANSFORMER ##

# this is one head

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()

    self.head_size = head_size
    self.key = nn.Linear(vocab_size, head_size)
    self.query = nn.Linear(vocab_size, head_size)
    self.value = nn.Linear(vocab_size, head_size)
    self.register_buffer('tril', torch.tril(T, T))

    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)

    wei = q @ k.transpose(-2, -1)
    tril = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = nn.functional.softmax(wei, dim=-1)  # approximate along channels
    wei = self.dropout(wei)

    out = wei @ v
    return out


# this is multiple heads

class MultiHeadAttention(nn.Module):
  def __init__(self, head_size, num_heads, n_embd):
    super().__init__()

    self.heads = nn.ModuleList([Head(head_size) for h in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(0.2)

  def forward(self,x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))

    return out


# this is a block

class Block(nn.Module):
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head

    self.self_attention = MultiHeadAttention(head_size, n_head, n_embd)
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4*n_embd),
        nn.ReLU(),
        nn.Linear(4*n_embd, n_embd),
        nn.Dropout(0.2)
    )

    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    # we do x + ... because we want to add residual connections
    # to make sure the net is learning full features
    x = x + self.self_attention(self.ln1(x))
    x = x + self.net(self.ln2(x))

    return x



# while encoding indices, we encode the data of each index, and its position.
# and then before predicting on it, we add both encodings.

class Transformer(nn.Module):
  def __init__(self, n_embd=32):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, n_embd)
    self.positional_embd = nn.Embedding(T, n_embd)

    n_head = 16
    n_layer = 10

    self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    # first we calculate encodings of data
    data_encodings = self.embedding(idx)

    # then we calculate encodings of position

    positional_encodings = self.positional_embd(torch.arange(T))
    total_encodings = positional_encodings + data_encodings # B,T,n_embd

    x = self.blocks(total_encodings)
    x = self.ln_f(x)
    x = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = x.shape
      x = x.view(B*T, C)
      targets = targets.view(B*T, C)
      loss = F.cross_entroy(targets, x)

    return x, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self.forward(idx)
      logits = logits[:, -1, :]  # get the last term predicted because thats the generated part
      logits = nn.functional.softmax(logits, dim=-1) # softmax along the one hot vector 
      idx_next = torch.multinomial(logits, num_samples=1) # form a distribution from softmaxed vector and sample from the formed distribution
      idx = torch.cat([idx, idx_next], dim=1)
    return idx
