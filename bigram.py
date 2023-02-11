import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# here are all the unique characters that occur in this text
char = sorted(list(set(text)))
vocab_size = len(char)

# create a mapping from characters to integers
ctoi = {ch:i for i,ch in enumerate(char)}
itoc = {i:ch for i,ch in enumerate(char)}
encode = lambda s:[ctoi[c] for c in s]
decode = lambda l:''.join(itoc[c] for c in l)

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train = data[:n]
val = data[n:]

# data loading
def get_batch(split):
  # generate a small batch of data of inputs x and targets y
  data = train if split == 'train' else val
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x,y

@torch.no_grad()
def estimate_loss():
  out = {}
  m.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X,Y = get_batch(split)
      logits, loss = m(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  m.train()
  return out
 
# super simple bigram model
class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self,inputs, targets=None):
    logits = self.token_embedding_table(inputs)
    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape   ## B-batch_size, T-time(seq), C-channel(vocab_size)
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self,inputs, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self(inputs)  # get the prediction
      logits = logits[:,-1,:]   # focus only on last time step
      probs = F.softmax(logits, dim=1)  # apply softmax to get probabilities
      inputs_next = torch.multinomial(probs , num_samples=1)  # (B,1)
      inputs = torch.cat((inputs, inputs_next), dim=1)  # (B,T+1)
    return inputs
  
m = BigramLanguageModel(vocab_size)
# m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for steps in range(max_iters):
  # every once in a while evaluate the loss on train and val sets
  if steps % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  # sample a batch of data
  xb,yb = get_batch('train')
  # evaluate loss  
  logits, loss = m(xb,yb)  
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

# generate from the model
inputs = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(inputs, max_new_tokens=300)[0].tolist()))



