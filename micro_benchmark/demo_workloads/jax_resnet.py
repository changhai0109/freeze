import jax
import jax.numpy as jnp
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer, FlaxResNetModel
import numpy as np

print(f"JAX Devices: {jax.devices()}")

def run_jax_resnet():
    print("--- Running JAX ResNet-50 ---")
    model = FlaxResNetModel.from_pretrained("microsoft/resnet-50")
    
    dummy_input = jnp.ones((1, 3, 224, 224))
    
    @jax.jit
    def inference(inputs):
        return model(pixel_values=inputs).logits
    
    print("Start execution loop...")
    for _ in range(10):
        _ = inference(dummy_input).block_until_ready()
    print("Done.")

if __name__ == "__main__":
    run_jax_resnet()
