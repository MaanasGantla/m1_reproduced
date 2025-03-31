# UCSC-VLAA/m1-32B-1K
# UCSC-VLAA/m1-7B-1K
# UCSC-VLAA/m1-7B-23K

limit ?= -1
overwrite ?= False
seed ?= 42
temperature ?= 0.0
thinking_budget_for_keep_think ?= 2048

all: \
thinking_budget \
thinking_budget_X-keep_think_below_budget

thinking_bugdet: \
m1-32b-1k-thinking_budget \
m1-7b-1k-thinking_budget \
m1-7b-23k-thinking_budget

thinking_budget_X-keep_think_below_budget: \
m1-32b-1k-thinking_budget_X-keep_think_below_budget \
m1-7b-1k-thinking_budget_X-keep_think_below_budget \
m1-7b-23k-thinking_budget_X-keep_think_below_budget

m1-32b-1k-thinking_budget:
	make -f exp/250319-ablation/thinking_budget.makefile \
	model_path=UCSC-VLAA/m1-32B-1K \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250319-ablation-thinking_budget/m1-32B-1K \
	tp=2 \
	dp=4 \
	seed=${seed} \
	temperature=${temperature}

m1-32b-1k-thinking_budget_X-keep_think_below_budget:
	make -f exp/250319-ablation/keep_thinking_below_budget.makefile \
	model_path=UCSC-VLAA/m1-32B-1K \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250319-ablation-keep_think_below_budget/thinking_budget_${thinking_budget_for_keep_think}/m1-32B-1K \
	thinking_budget=${thinking_budget_for_keep_think} \
	tp=2 \
	dp=4 \
	seed=${seed} \
	temperature=${temperature}


m1-7b-1k-thinking_budget:
	make -f exp/250319-ablation/thinking_budget.makefile \
	model_path=UCSC-VLAA/m1-7B-1K \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250319-ablation-thinking_budget/m1-7B-1K \
	tp=1 \
	dp=8 \
	seed=${seed} \
	temperature=${temperature}

m1-7b-1k-thinking_budget_X-keep_think_below_budget:
	make -f exp/250319-ablation/keep_thinking_below_budget.makefile \
	model_path=UCSC-VLAA/m1-7B-1K \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250319-ablation-keep_think_below_budget/thinking_budget_${thinking_budget_for_keep_think}/m1-7B-1K \
	thinking_budget=${thinking_budget_for_keep_think} \
	tp=1 \
	dp=8 \
	seed=${seed} \
	temperature=${temperature}


m1-7b-23k-thinking_budget:
	make -f exp/250319-ablation/thinking_budget.makefile \
	model_path=UCSC-VLAA/m1-7B-23K \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250319-ablation-thinking_budget/m1-7B-23K \
	tp=1 \
	dp=8 \
	seed=${seed} \
	temperature=${temperature}

m1-7b-23k-thinking_budget_X-keep_think_below_budget:
	make -f exp/250319-ablation/keep_thinking_below_budget.makefile \
	model_path=UCSC-VLAA/m1-7B-23K \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250319-ablation-keep_think_below_budget/thinking_budget_${thinking_budget_for_keep_think}/m1-7B-23K \
	thinking_budget=${thinking_budget_for_keep_think} \
	tp=1 \
	dp=8 \
	seed=${seed} \
	temperature=${temperature}