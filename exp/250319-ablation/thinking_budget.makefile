limit ?= -1
overwrite ?= False
tp ?= 1
dp ?= 4
model_path ?= UCSC-VLAA/m1-32B-1K
output_dir ?= outputs/250319-ablation-thinking_budget/m1-32B-1K
batch_size ?= 1024
seed ?= 42
temperature ?= 0.0


all: start_server 8192 4096 2048 1024 512 256 128 stop_server

start_server:
	@echo "---------------------------------------------------------------------"
	@echo ">>> DIAGNOSTIC (thinking_budget.makefile for start_server):"
	@echo ">>>   About to call sub-make (template.makefile for eval_llm-start_server)."
	@echo ">>>   Current values in thinking_budget.makefile:"
	@echo ">>>     model_path='${model_path}'"
	@echo ">>>     tp='${tp}'"  # Should be 1 from your main command
	@echo ">>>     dp='${dp}'"  # Should be 4 from your main command
	@echo "---------------------------------------------------------------------"
	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=${model_path} \
	exp_name=thinking_budget \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	tp=${tp} \
	dp=${dp} \
	eval_llm-start_server \
	seed=${seed} \
	temperature=${temperature}

128:
	make -f exp/250318-eval-medical_llm/template.makefile \
	max_new_tokens=128 \
	exp_name=thinking_buget_128 \
	force_think=True \
	keep_think_below_budget_times=0 \
	batch_size=${batch_size} \
	model_path=${model_path} \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	dp=${dp}\
	tp=${tp}\
	eval_llm-only_inference \
	seed=${seed} \
	temperature=${temperature}

256:
	make -f exp/250318-eval-medical_llm/template.makefile \
	max_new_tokens=256 \
	exp_name=thinking_buget_256 \
	force_think=True \
	keep_think_below_budget_times=0 \
	batch_size=${batch_size} \
	model_path=${model_path} \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	dp=${dp}\
	tp=${tp}\
	eval_llm-only_inference \
	seed=${seed} \
	temperature=${temperature}

512:
	make -f exp/250318-eval-medical_llm/template.makefile \
	max_new_tokens=512 \
	exp_name=thinking_buget_512 \
	force_think=True \
	keep_think_below_budget_times=0 \
	batch_size=${batch_size} \
	model_path=${model_path} \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	dp=${dp}\
	tp=${tp}\
	eval_llm-only_inference \
	seed=${seed} \
	temperature=${temperature}

1024:
	make -f exp/250318-eval-medical_llm/template.makefile \
	max_new_tokens=1024 \
	exp_name=thinking_buget_1024 \
	force_think=True \
	keep_think_below_budget_times=0 \
	batch_size=${batch_size} \
	model_path=${model_path} \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	dp=${dp}\
	tp=${tp}\
	eval_llm-only_inference \
	seed=${seed} \
	temperature=${temperature}

2048:
	make -f exp/250318-eval-medical_llm/template.makefile \
	max_new_tokens=2048 \
	exp_name=thinking_buget_2048 \
	force_think=True \
	keep_think_below_budget_times=0 \
	batch_size=${batch_size} \
	model_path=${model_path} \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	dp=${dp}\
	tp=${tp}\
	eval_llm-only_inference \
	seed=${seed} \
	temperature=${temperature}

4096:
	make -f exp/250318-eval-medical_llm/template.makefile \
	max_new_tokens=4096 \
	exp_name=thinking_buget_4096 \
	force_think=True \
	keep_think_below_budget_times=0 \
	batch_size=${batch_size} \
	model_path=${model_path} \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	dp=${dp}\
	tp=${tp}\
	eval_llm-only_inference \
	seed=${seed} \
	temperature=${temperature}

8192:
	make -f exp/250318-eval-medical_llm/template.makefile \
	max_new_tokens=8192 \
	exp_name=thinking_buget_8192 \
	force_think=True \
	keep_think_below_budget_times=0 \
	batch_size=${batch_size} \
	model_path=${model_path} \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	dp=${dp}\
	tp=${tp}\
	eval_llm-only_inference \
	seed=${seed} \
	temperature=${temperature}

12288:
	make -f exp/250318-eval-medical_llm/template.makefile \
	max_new_tokens=12288 \
	exp_name=thinking_buget_12288 \
	force_think=True \
	keep_think_below_budget_times=0 \
	batch_size=${batch_size} \
	model_path=${model_path} \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	dp=${dp}\
	tp=${tp}\
	eval_llm-only_inference \
	seed=${seed} \
	temperature=${temperature}

16384:
	make -f exp/250318-eval-medical_llm/template.makefile \
	max_new_tokens=16384 \
	exp_name=thinking_buget_16384 \
	force_think=True \
	keep_think_below_budget_times=0 \
	batch_size=${batch_size} \
	model_path=${model_path} \
	output_dir=${output_dir} \
	limit=${limit} \
	overwrite=${overwrite} \
	dp=${dp}\
	tp=${tp}\
	eval_llm-only_inference \
	seed=${seed} \
	temperature=${temperature}

# ignore error with - at the beginning
stop_server:
	-pgrep 'sglang' -f | xargs kill -9
	sleep 10
	@echo "return code $?"