import torch
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import contextlib

# ================= CONFIGURATION =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ITERATIONS = 20  # Matches your paper text
WARMUP_STEPS = 5

# ================= WORKLOAD DEFINITIONS =================


class WorkloadManager:
    def __init__(self):
        self.device = DEVICE
        print(f"Loading workloads on {self.device}...")

        # 1. Computer Vision: ResNet-50
        self.resnet = models.resnet50().to(self.device)
        self.resnet_input = torch.randn(16, 3, 224, 224).to(
            self.device
        )  # Batch size 16

        # 2. LLM: Llama-3.2-1B
        # Note: You need access to this model on HuggingFace.
        # If not, swap for "gpt2" for testing.
        try:
            model_id = "meta-llama/Llama-3.2-1B"
            self.llama = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to(self.device)
            self.llama_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llama_input = self.llama_tokenizer(
                "Hello, how are you?", return_tensors="pt"
            ).to(self.device)
        except Exception as e:
            print(
                f"Warning: Could not load Llama (check login/internet). Skipping LLM. Error: {e}"
            )
            self.llama = None

        # 3. Scientific: Game of Life (Vectorized implementation)
        self.gol_grid = (torch.rand(1024, 1024) > 0.5).float().to(self.device)
        self.gol_kernel = (
            torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            .float()
            .to(self.device)
            .view(1, 1, 3, 3)
        )

    def run_resnet_inference(self):
        with torch.no_grad():
            _ = self.resnet(self.resnet_input)

    def run_resnet_training(self):
        # Fake training step
        self.resnet.train()
        output = self.resnet(self.resnet_input)
        loss = output.sum()
        loss.backward()
        self.resnet.zero_grad()

    def run_llama_inference(self):
        if self.llama:
            with torch.no_grad():
                # Generate 10 new tokens
                _ = self.llama.generate(**self.llama_input, max_new_tokens=10)

    def run_game_of_life(self):
        # Vectorized step using Conv2d to count neighbors
        # Pad input to handle edges
        for _ in range(200):
            padded = F.pad(
                self.gol_grid.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="circular"
            )
            neighbors = F.conv2d(padded, self.gol_kernel)

            # Game of Life Rules:
            # 1. Alive (1) & 2 or 3 neighbors -> Alive
            # 2. Dead (0) & 3 neighbors -> Alive
            # 3. Else -> Dead
            current = self.gol_grid.unsqueeze(0).unsqueeze(0)
            alive_cond = (current == 1) & ((neighbors == 2) | (neighbors == 3))
            born_cond = (current == 0) & (neighbors == 3)

            self.gol_grid = (alive_cond | born_cond).float().squeeze()


# ================= BENCHMARK ENGINE =================


def measure_overhead(workload_func, mode="native", run_name="Experiment"):
    # Warmup
    for _ in range(WARMUP_STEPS):
        workload_func()
    torch.cuda.synchronize()

    start_time = time.time()

    # --- EXECUTION MODES ---
    if mode == "native":
        # Baseline: Pure execution
        for _ in range(NUM_ITERATIONS):
            workload_func()

    elif mode == "kineto":
        # Baseline: PyTorch Kineto (Standard Profiler)
        # We trace typical activities (CPU + CUDA)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,  # Set True if you want deeper stack tracing (higher overhead)
        ) as prof:
            for _ in range(NUM_ITERATIONS):
                workload_func()
                prof.step()
    elif mode == "kineto_gpu":
        # Baseline: PyTorch Kineto (Standard Profiler)
        # We trace typical activities (CPU + CUDA)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,  # Set True if you want deeper stack tracing (higher overhead)
        ) as prof:
            for _ in range(NUM_ITERATIONS):
                workload_func()
                prof.step()

    # Placeholder for your tool
    elif mode == "sys_timeline":
        # TODO: Inject your tool's start command here
        # e.g., sniper.start_trace()
        for _ in range(NUM_ITERATIONS):
            workload_func()
        # TODO: Inject your tool's stop command here

    # -----------------------

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency = (total_time / NUM_ITERATIONS) * 1000  # ms

    print(
        f"[{run_name}] Mode: {mode:<10} | Total: {total_time:.4f}s | Avg/Iter: {avg_latency:.2f}ms"
    )
    return total_time


# ================= MAIN =================

if __name__ == "__main__":
    wm = WorkloadManager()

    tasks = [
        ("ResNet Inf", wm.run_resnet_inference),
        ("ResNet Train", wm.run_resnet_training),
        ("Game of Life", wm.run_game_of_life),
        ("Llama Inf", wm.run_llama_inference),  # Uncomment if Llama is loaded
    ]

    results = {}

    print(f"\nStarting Benchmark ({NUM_ITERATIONS} iterations per run)...\n")

    for task_name, task_func in tasks:
        print(f"--- Benchmarking {task_name} ---")

        # 1. Native
        t_native = measure_overhead(task_func, mode="native", run_name=task_name)

        # # 2. Kineto
        # t_kineto = measure_overhead(task_func, mode="kineto", run_name=task_name)

        # # 3. Kineto GPU Only
        # t_kineto_gpu = measure_overhead(
        #     task_func, mode="kineto_gpu", run_name=task_name
        # )

        # # Calculate Overhead
        # overhead_pct = ((t_kineto - t_native) / t_native) * 100
        # overhead_gpu_pct = ((t_kineto_gpu - t_native) / t_native) * 100
        # print(f">>> Kineto Overhead: +{overhead_pct:.2f}%")
        # print(f">>> Kineto GPU Only Overhead: +{overhead_gpu_pct:.2f}%\n")
