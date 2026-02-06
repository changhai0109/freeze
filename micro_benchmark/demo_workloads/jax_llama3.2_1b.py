import jax
import jax.numpy as jnp
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer, FlaxResNetModel
import numpy as np

print(f"JAX Devices: {jax.devices()}")

def run_jax_llama():
    print("\n--- Running JAX Llama-3.2-1B ---")
    model_id = "meta-llama/Llama-3.2-1B"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = FlaxAutoModelForCausalLM.from_pretrained(model_id, dtype=jnp.float16)
    except Exception as e:
        print(f"Error loading Llama: {e}")
        return

    prompt = "Hello JAX."
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    
    print("Start execution loop...")
    for _ in range(5):
        model.generate(input_ids, max_new_tokens=20, do_sample=False).sequences.block_until_ready()
    print("Done.")

if __name__ == "__main__":
    run_jax_llama()
