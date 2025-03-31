limit ?= -1
overwrite ?= False
tp ?= 1
dp ?= 8
model_path ?= UCSC-VLAA/m1-32B-1K
output_dir ?= outputs/seed_42-temperature_0.0/250319-ablation-keep_thinking_below_budget-8192/m1-32B-1K
batch_size ?= 1024
thinking_budget ?= 8192
seed ?= 42
temperature ?= 0.0


all: start_server 0 1 2 4 6 8 stop_server

start_server:
	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=${model_path} \
	exp_name=keep_thinking_below_budget \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	tp=${tp} \
	dp=${dp} \
	eval_llm-start_server

0:
	make -f exp/250318-eval-medical_llm/template.makefile \
	max_new_tokens=${thinking_budget} \
	exp_name=keep_thinking_below_budget_0 \
	force_think=True \
	keep_think_below_budget_times=0 \
	batch_size=${batch_size} \
	model_path=${model_path} \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	eval_llm-only_inference \
	seed=${seed} \
	temperature=${temperature}

1:
	make -f exp/250318-eval-medical_llm/template.makefile \
	max_new_tokens=${thinking_budget} \
	exp_name=keep_thinking_below_budget_1 \
	force_think=True \
	keep_think_below_budget_times=1 \
	batch_size=${batch_size} \
	model_path=${model_path} \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	eval_llm-only_inference \
	seed=${seed} \
	temperature=${temperature}

2:
	make -f exp/250318-eval-medical_llm/template.makefile \
	max_new_tokens=${thinking_budget} \
	exp_name=keep_thinking_below_budget_2 \
	force_think=True \
	keep_think_below_budget_times=2 \
	batch_size=${batch_size} \
	model_path=${model_path} \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	eval_llm-only_inference \
	seed=${seed} \
	temperature=${temperature}

4:
	make -f exp/250318-eval-medical_llm/template.makefile \
	max_new_tokens=${thinking_budget} \
	exp_name=keep_thinking_below_budget_4 \
	force_think=True \
	keep_think_below_budget_times=4 \
	batch_size=${batch_size} \
	model_path=${model_path} \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	eval_llm-only_inference \
	seed=${seed} \
	temperature=${temperature}

6:
	make -f exp/250318-eval-medical_llm/template.makefile \
	max_new_tokens=${thinking_budget} \
	exp_name=keep_thinking_below_budget_6 \
	force_think=True \
	keep_think_below_budget_times=6 \
	batch_size=${batch_size} \
	model_path=${model_path} \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	eval_llm-only_inference \
	seed=${seed} \
	temperature=${temperature}

8:
	make -f exp/250318-eval-medical_llm/template.makefile \
	max_new_tokens=${thinking_budget} \
	exp_name=keep_thinking_below_budget_8 \
	force_think=True \
	keep_think_below_budget_times=8 \
	batch_size=${batch_size} \
	model_path=${model_path} \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	eval_llm-only_inference \
	seed=${seed} \
	temperature=${temperature}


# ignore error with - at the beginning
stop_server:
	-pgrep 'sglang' -f | xargs kill -9
	sleep 10
	@echo "return code $?"