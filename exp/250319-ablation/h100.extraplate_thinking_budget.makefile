# UCSC-VLAA/m1-32B-1K
# UCSC-VLAA/m1-7B-1K
# UCSC-VLAA/m1-7B-23K

limit ?= -1
overwrite ?= False
seed ?= 42
temperature ?= 0.0

all: \
m1-32b-1k-thinking_budget \
m1-7b-1k-thinking_budget \
m1-7b-23k-thinking_budget

m1-32b-1k-thinking_budget:
	make -f exp/250319-ablation/thinking_budget.makefile \
	model_path=UCSC-VLAA/m1-32B-1K \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250319-ablation-thinking_budget/m1-32B-1K \
	tp=2 \
	dp=4 \
	seed=${seed} \
	temperature=${temperature} \
	start_server 16384 12288 stop_server


m1-7b-1k-thinking_budget:
	make -f exp/250319-ablation/thinking_budget.makefile \
	model_path=UCSC-VLAA/m1-7B-1K \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250319-ablation-thinking_budget/m1-7B-1K \
	tp=1 \
	dp=8 \
	seed=${seed} \
	temperature=${temperature} \
	start_server 16384 12288 stop_server


m1-7b-23k-thinking_budget:
	make -f exp/250319-ablation/thinking_budget.makefile \
	model_path=UCSC-VLAA/m1-7B-23K \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250319-ablation-thinking_budget/m1-7B-23K \
	tp=1 \
	dp=8 \
	seed=${seed} \
	temperature=${temperature} \
	start_server 16384 12288 stop_server

