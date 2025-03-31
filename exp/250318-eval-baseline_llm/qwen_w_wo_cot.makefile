tp ?= 1
dp ?= 8
max_new_tokens ?= 2048
batch_size ?= 1024
temperature ?= 0.0
limit ?= -1
overwrite ?= False
model_path ?= Qwen/Qwen2.5-7B-Instruct
exp_name ?= Qwen2.5-7B-Instruct
output_dir ?= outputs/eval_baseline_llm/ 
seed ?= 42
timeout ?= 1800

all: eval_baseline_llm

eval_baseline_llm: eval_baseline_llm-start_server eval_baseline_llm-only_inference eval_baseline_llm-only_inference-cot stop_server

eval_baseline_llm-start_server:
	@echo "Evaluating LLM baselines"
	@echo "model_path=${model_path}"
	@echo "tp=${tp}"
	@echo "dp=${dp}"

	python src/eval/inference_keep_think.py -u "model_path=${model_path},tp=${tp},dp=${dp}" --only_start_server

eval_baseline_llm-only_inference:
	@echo "model_path=${model_path}"
	@echo "max_new_tokens=${max_new_tokens}"
	@echo "batch_size=${batch_size}"
	@echo "suffix_prompt=${suffix_prompt}"
	@echo "temperature=${temperature}"
	@echo "limit=${limit}"
	@echo "overwrite=${overwrite}"
	@echo "seed=${seed}"
	@echo "output_dir=${output_dir}"
	@echo "timeout=${timeout}"

	python src/eval/inference_keep_think.py -u \
	"\
	suffix_prompt=Return your final response within \\boxed{{}}.,\
	output_dir=${output_dir},\
	model_path=${model_path},\
	exp_name=${exp_name},\
	max_new_tokens=${max_new_tokens},\
	batch_size=${batch_size},\
	temperature=${temperature},\
	limit=${limit},\
	overwrite=${overwrite},\
	timeout=${timeout},\
	seed=${seed}\
	" \
	--only_inference

eval_baseline_llm-only_inference-cot:
	@echo "model_path=${model_path}"
	@echo "max_new_tokens=${max_new_tokens}"
	@echo "batch_size=${batch_size}"
	@echo "suffix_prompt=${suffix_prompt}"
	@echo "temperature=${temperature}"
	@echo "limit=${limit}"
	@echo "overwrite=${overwrite}"
	@echo "seed=${seed}"
	@echo "output_dir=${output_dir}"
	@echo "timeout=${timeout}"

	python src/eval/inference_keep_think.py -u \
	"\
	suffix_prompt=Let's think step by step. Return your final response within \\boxed{{}}.,\
	exp_name=${exp_name}-cot,\
	model_path=${model_path},\
	output_dir=${output_dir},\
	max_new_tokens=${max_new_tokens},\
	batch_size=${batch_size},\
	temperature=${temperature},\
	limit=${limit},\
	overwrite=${overwrite},\
	timeout=${timeout},\
	seed=${seed}\
	" \
	--only_inference

# ignore error with - at the beginning
stop_server:
	-pgrep 'sglang' -f | xargs kill -9
	sleep 10
	@echo "return code $?"