#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:11:02 2023

@author: kyletrocki
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

file_name = "../books/1_The_Philosophers_Stone.txt"

with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)

# Develop way to tokenize my input:
stoi = {ch: i for i,ch in enumerate(chars)} # Make dictionary with key as char and value as index
itos = {i: ch for i,ch in enumerate(chars)} # Make dictionary with key as index and value as char

encode = lambda s: [stoi[c] for c in s] # take strting and get list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # take strting and get list of ints

# Encode the entire dataset as one long string
data = torch.tensor(encode(text), dtype=torch.long)


n= int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8 # size of chunks used to train
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+ 1]


for t in range(block_size):
    context = x[:t+1]
    target = y[t]
   # print(f"when input is {context} target is {target}")
    
    
torch.manual_seed(1337)
batch_size = 4 # Num of sequences in parallel
block_size = 8 # Max context length for predicitons

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # Shifted by 1 position
    return x, y


xb, yb = get_batch('train')

for b in range(batch_size):
    for t in range(block_size):
        context =xb[b, :t+1]
        target = yb[b,t]
        #print(f"when input is {context.tolist()} target is {target}")



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
        
    def forward(self, idx, targets=None):
        # Here index (idx) and the targets are (B,T) tensor of integers
        logits = self.token_embedding_table(idx) 
        # Torch will arrange as batch, time, channel (B, T, C) Tensor
        # we extract out the logits (row in the vocab_size by vocab size tensor)
        # this scores them based on the next character prediction
        
        # This makes sense because its a token by token basis- this means each token is only seeing itself
        
        
        
        # Now we want to evaluate the prediction and see how good it was
        # for this we are going to evaluate the loss function:
        # For this we will use the negative log likelihood loss (cross_entropy in torch)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx) # Get predictions
            logits = logits[:, -1, :] #Only focus on last element in time dimension since that predicts what comes next- becomes (B,C)
            probs = F.softmax(logits, dim=-1) #(B,C)
            idx_next = torch.multinomial(probs, num_samples=1)# Get the best prediction- (B,1)
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)

print(logits)
print(loss)

idx = torch.zeros((1,1), dtype=torch.long) # Feeding in 0 as first char- new line char so makes sense
print(decode(m.generate(idx= idx, max_new_tokens=100)[0].tolist())) #m.generate will continue the sequence - works on the batch level


#up this point the model returns junk because it has yet to be trained:
    
#so lets train it- small model so we can have a large learning rate
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10000):
    #sample a batch of data:
    xb, yb = get_batch('trian')
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True) # Zero the gradients from the previous step
    loss.backward() # get gradients for the params
    optimizer.step() # now update the parameters

print(loss.item())

print(decode(m.generate(idx= idx, max_new_tokens=500)[0].tolist()))

# Now we need to add some context 
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BigramLanguageModel(vocab_size)
m = model.to(device)

def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out







