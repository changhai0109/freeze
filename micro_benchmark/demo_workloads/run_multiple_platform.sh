# CUDA_VISIBLE_DEVICES=0 NO_EAGER_LOAD=1 NOTRACE_DUMP_CHROME_TRACE=./torch_resnet.json LD_PRELOAD=../../notrace/notrace.so python torch_resnet.py > torch_resnet.stdout 2>&1
# CUDA_VISIBLE_DEVICES=0 NO_EAGER_LOAD=1 NOTRACE_DUMP_CHROME_TRACE=./torch_llama.json LD_PRELOAD=../../notrace/notrace.so python torch_llama3.2_1b.py > torch_llama.stdout 2>&1
CUDA_VISIBLE_DEVICES=0 NO_EAGER_LOAD=1 NOTRACE_DUMP_CHROME_TRACE=./jax_resnet.json LD_PRELOAD=../../notrace/notrace.so python jax_resnet.py > jax_resnet.stdout 2>&1
CUDA_VISIBLE_DEVICES=0 NO_EAGER_LOAD=1 NOTRACE_DUMP_CHROME_TRACE=./jax_llama.json LD_PRELOAD=../../notrace/notrace.so python jax_llama3.2_1b.py > jax_llama.stdout 2>&1
