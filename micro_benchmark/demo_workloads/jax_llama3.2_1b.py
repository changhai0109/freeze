from xml.parsers.expat import model
import jax
import jax.numpy as jnp
from transformers import (
    FlaxAutoModelForCausalLM,
    AutoTokenizer,
    FlaxResNetModel,
    GenerationConfig,
    AutoConfig,
)
import numpy as np

print(f"JAX Devices: {jax.devices()}")


def run_jax_llama():
    print("\n--- Running JAX Llama-3.2-1B ---")
    model_id = "meta-llama/Llama-3.2-1B"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
        config.max_position_embeddings = 4096
        config.use_cache = True
        print(f"Model config: {config}")
        model = FlaxAutoModelForCausalLM.from_pretrained(
            model_id, config=config, dtype=jnp.float16
        )
    except Exception as e:
        print(f"Error loading Llama: {e}")
        return

    prompt = "Hello JAX."
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]

    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_length = int(input_ids.shape[1] + 20)
    gen_cfg.do_sample = False
    gen_cfg.pad_token_id = tokenizer.eos_token_id
    gen_cfg.eos_token_id = tokenizer.eos_token_id

    print("Start execution loop...")
    for _ in range(5):
        model.generate(
            input_ids, generation_config=gen_cfg
        ).sequences.block_until_ready()
    print("Done.")


if __name__ == "__main__":
    run_jax_llama()
