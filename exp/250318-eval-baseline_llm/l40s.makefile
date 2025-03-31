# Qwen/Qwen2.5-7B-Instruct
# Qwen/Qwen2.5-32B-Instruct
# Qwen/Qwen2.5-72B-Instruct

limit ?= -1
overwrite ?= False
seed ?= 42
temperature ?= 0.0

all: \
7b \
32b \
72b


7b:
	make -f exp/250318-eval-baseline_llm/qwen_w_wo_cot.makefile \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-baseline_llm/ \
	model_path=Qwen/Qwen2.5-7B-Instruct \
	exp_name=Qwen2.5-7B-Instruct \
	limit=${limit} \
	overwrite=${overwrite} \
	seed=${seed} \
	temperature=${temperature}

32b:
	make -f exp/250318-eval-baseline_llm/qwen_w_wo_cot.makefile \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-baseline_llm/ \
	model_path=Qwen/Qwen2.5-32B-Instruct \
	exp_name=Qwen2.5-32B-Instruct \
	limit=${limit} \
	overwrite=${overwrite} \
	tp=4 \
	dp=2 \
	seed=${seed} \
	temperature=${temperature}

72b:
	make -f exp/250318-eval-baseline_llm/qwen_w_wo_cot.makefile \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-baseline_llm/ \
	model_path=Qwen/Qwen2.5-72B-Instruct \
	exp_name=Qwen2.5-72B-Instruct \
	limit=${limit} \
	overwrite=${overwrite} \
	tp=8 \
	dp=1 \
	seed=${seed} \
	temperature=${temperature}