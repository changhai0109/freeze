import torch, gc

a = torch.empty(2000, 3000)
b = torch.empty(3000, 5000)
c = torch.empty(2000, 5000)

a = a.cuda()
b = b.cuda()
c = c.cuda()

with torch.no_grad():
    y = torch.matmul(a, b)+c
    y = torch.matmul(a, b)+c

del a, b, c
gc.collect()

torch.cuda.synchronize()
torch.cuda.empty_cache()

ycpu = y.cpu()
del y
gc.collect()

torch.cuda.synchronize()
torch.cuda.empty_cache()

