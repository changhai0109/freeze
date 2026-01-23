import torch
import gc

if torch.cuda.is_available():
    # Force CUDA context initialization if it hasn't happened implicitly
    # by just importing torch.cuda. We could also just call torch.zeros(1).cuda()
    torch.cuda.current_device()
    print("CUDA is available and initialized.")
else:
    print("CUDA not available.")
    
a = torch.empty(2000, 3000)
a = a.cuda()
a = a * a
del a
gc.collect()

# It's good practice to ensure everything is synchronized and cached cleared
# even for a minimal script, though it likely won't affect these specific allocations.
torch.cuda.synchronize()
torch.cuda.empty_cache()

print("Python script finishing.")
