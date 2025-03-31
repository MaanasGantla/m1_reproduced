# mmqm/m1-7b-random_1k-epoch_5
# mmqm/m1-7b-hard_random_1k-epoch_5
# mmqm/m1-7b-domain_1k-epoch_5
# mmqm/m1-7b-random_23k-epoch_5

limit ?= -1
overwrite ?= False
seed ?= 42
temperature ?= 0.0

all: \
random_1k \
hard_random_1k \
domain_1k \
random_23k

random_1k:
	make -f exp/250319-ablation/thinking_budget.makefile \
	model_path=mmqm/m1-7b-random_1k-epoch_5 \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250319-ablation-curation/random_1k \
	tp=2 \
	dp=4 \
	seed=${seed} \
	temperature=${temperature} \
	start_server 2048 stop_server

hard_random_1k:
	make -f exp/250319-ablation/thinking_budget.makefile \
	model_path=mmqm/m1-7b-hard_random_1k-epoch_5 \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250319-ablation-curation/hard_random_1k \
	tp=2 \
	dp=4 \
	seed=${seed} \
	temperature=${temperature} \
	start_server 2048 stop_server

domain_1k:
	make -f exp/250319-ablation/thinking_budget.makefile \
	model_path=mmqm/m1-7b-domain_1k-epoch_5 \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250319-ablation-curation/domain_1k \
	tp=2 \
	dp=4 \
	seed=${seed} \
	temperature=${temperature} \
	start_server 2048 stop_server

random_23k:
	make -f exp/250319-ablation/thinking_budget.makefile \
	model_path=mmqm/m1-7b-random_23k-epoch_5 \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250319-ablation-curation/random_23k \
	tp=2 \
	dp=4 \
	seed=${seed} \
	temperature=${temperature} \
	start_server 2048 stop_server