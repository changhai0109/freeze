import torch
import torchvision.models as models
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

def run_llama():
    print("\n--- Running PyTorch Llama-3.2-1B ---")
    model_id = "meta-llama/Llama-3.2-1B"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading Llama: {e}")
        return

    prompt = "Hello, I am a research assistant."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("Start execution loop...")
    for i in range(5):
        _ = model.generate(**inputs, max_new_tokens=20)
        torch.cuda.synchronize()
    print("Done.")

if __name__ == "__main__":
    with torch.no_grad():
        # run_resnet()
        run_llama()
