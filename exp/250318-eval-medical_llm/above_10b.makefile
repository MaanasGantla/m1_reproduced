# use h100
tp ?= 4
dp ?= 2
limit ?= -1
overwrite ?= False
seed ?= 42
temperature ?= 0.0

all: \
Med42 \
OpenBioLLM \
UltraMedical \
HuatuoGPT-o1

# m42-health/med42-70b is llama 2, which only has 2048 context length, so we remove it.
Med42:
	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=m42-health/Llama3-Med42-70B \
	exp_name=Llama3-Med42-70B \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	dp=${dp} \
	tp=${tp} \
	seed=${seed} \
	temperature=${temperature}


# aaditya/Llama3-OpenBioLLM-70B does not have chat template
OpenBioLLM:
	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=aaditya/Llama3-OpenBioLLM-70B \
	tokenizer_path=meta-llama/Meta-Llama-3-8B-Instruct \
	exp_name=Llama3-OpenBioLLM-70B \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	dp=${dp} \
	tp=${tp} \
	seed=${seed} \
	temperature=${temperature}



UltraMedical:
	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=TsinghuaC3I/Llama-3-70B-UltraMedical \
	exp_name=Llama-3-70B-UltraMedical \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	dp=${dp} \
	tp=${tp} \
	seed=${seed} \
	temperature=${temperature}


HuatuoGPT-o1:
	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=FreedomIntelligence/HuatuoGPT-o1-70B \
	exp_name=HuatuoGPT-o1-70B \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	dp=${dp} \
	tp=${tp} \
	seed=${seed} \
	temperature=${temperature}


	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=FreedomIntelligence/HuatuoGPT-o1-72B \
	exp_name=HuatuoGPT-o1-72B \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	dp=${dp} \
	tp=${tp} \
	seed=${seed} \
	temperature=${temperature}

