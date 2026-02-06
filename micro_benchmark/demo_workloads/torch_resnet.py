import torch
import torchvision.models as models
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

def run_resnet():
    print("--- Running PyTorch ResNet-50 ---")
    model = models.resnet50().to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    print("Start execution loop...")
    for i in range(10):
        output = model(dummy_input)
        torch.cuda.synchronize()
    print("Done.")

if __name__ == "__main__":
    with torch.no_grad():
        run_resnet()
