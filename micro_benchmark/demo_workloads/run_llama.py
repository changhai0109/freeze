import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# 1. 设置模型 ID (需要 HF 权限)
model_id = "meta-llama/Llama-3.2-1B-Instruct"

# 2. 检查 CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# 3. 加载 Tokenizer 和 模型
# 使用 bfloat16 或 float16 以节省显存并模拟真实推理场景
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map=device
)

# 4. 准备输入
prompt = "Write a hello world program in C++."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("--- Model Loaded. Starting Inference ---")

# Optional: 预热 (Warmup)
# 很多 CUDA Context 初始化和内存分配发生在这里，
# 如果只想 Trace 纯粹的计算 Kernel，通常会忽略这部分的 Trace。
print("Running warmup...")
_ = model.generate(**inputs, max_new_tokens=10)
torch.cuda.synchronize() # 等待预热完成

# 5. 正式运行 (这是你想用你的工具抓取的部分)
print("Running generation (Start Tracing)...")
start_time = time.time()

# 生成
outputs = model.generate(
    **inputs, 
    max_new_tokens=50,   # 生成 50 个 token
    do_sample=False      # 关闭采样，使用贪婪解码，减少随机性
)

torch.cuda.synchronize() # 确保 GPU 计算全部完成
end_time = time.time()

print(f"Inference finished in {end_time - start_time:.4f} seconds")

# 6. 解码输出
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nOutput:\n" + "-"*20)
print(response)
print("-" * 20)

