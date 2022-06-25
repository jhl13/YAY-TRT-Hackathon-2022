import torch
from time import time_ns

q = torch.randn((1089,6,10,64)).to("cuda")
k = torch.randn((1089,6,10,64)).to("cuda")

attn = (q @ k.transpose(-2, -1))

t0= time_ns()
for i in range(30):
    attn = (q @ k.transpose(-2, -1))
t1 = time_ns()
timePerInference = (t1-t0)/1000/1000/30
print(timePerInference)

q_ = q.view(-1, 10, 64)
k_ = k.transpose(-2, -1).contiguous().view(-1, 64, 10)
t0= time_ns()
for i in range(30):
    attn_ = (q_ @ k_)
t1 = time_ns()
timePerInference = (t1-t0)/1000/1000/30
print(timePerInference)
