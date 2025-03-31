limit ?= -1
overwrite ?= False
seed ?= 42
temperature ?= 0.0

all: \
MedLlama3 \
Med42 \
MMed\
OpenBioLLM \
UltraMedical \
HuatuoGPT-o1


# johnsnowlabs/JSL-MedLlama-3-8B-v2.0 does not have chat template
MedLlama3:
	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=johnsnowlabs/JSL-MedLlama-3-8B-v2.0 \
	tokenizer_path=meta-llama/Meta-Llama-3-8B-Instruct \
	exp_name=JSL-MedLlama-3-8B-v2.0 \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	seed=${seed} \
	temperature=${temperature}

	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=johnsnowlabs/JSL-MedLlama-3-8B-v1.0 \
	tokenizer_path=meta-llama/Meta-Llama-3-8B-Instruct \
	exp_name=JSL-MedLlama-3-8B-v1.0 \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	seed=${seed} \
	temperature=${temperature}


Med42:
	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=m42-health/Llama3-Med42-8B \
	exp_name=Llama3-Med42-8B \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	seed=${seed} \
	temperature=${temperature}


MMed:
	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=Henrychur/MMed-Llama-3-8B-EnIns \
	exp_name=MMed-Llama-3-8B-EnIns \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	seed=${seed} \
	temperature=${temperature}

	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=Henrychur/MMedS-Llama-3-8B \
	exp_name=MMedS-Llama-3-8B \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	seed=${seed} \
	temperature=${temperature}

	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=Henrychur/MMed-Llama-3-8B \
	exp_name=MMed-Llama-3-8B \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	seed=${seed} \
	temperature=${temperature}


# aaditya/Llama3-OpenBioLLM-8B does not have chat template
OpenBioLLM:
	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=aaditya/Llama3-OpenBioLLM-8B \
	tokenizer_path=meta-llama/Meta-Llama-3-8B-Instruct \
	exp_name=Llama3-OpenBioLLM-8B \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	seed=${seed} \
	temperature=${temperature}


UltraMedical:
	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=TsinghuaC3I/Llama-3.1-8B-UltraMedical \
	exp_name=Llama-3.1-8B-UltraMedical \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	seed=${seed} \
	temperature=${temperature}

	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=TsinghuaC3I/Llama-3-8B-UltraMedical \
	exp_name=Llama-3-8B-UltraMedical \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	seed=${seed} \
	temperature=${temperature}

HuatuoGPT-o1:
	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=FreedomIntelligence/HuatuoGPT-o1-7B \
	exp_name=HuatuoGPT-o1-7B \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	seed=${seed} \
	temperature=${temperature}

	make -f exp/250318-eval-medical_llm/template.makefile \
	model_path=FreedomIntelligence/HuatuoGPT-o1-8B \
	exp_name=HuatuoGPT-o1-8B \
	output_dir=outputs/seed_${seed}-temperature-${temperature}/250318-eval-medical_llm \
	limit=${limit} \
	overwrite=${overwrite} \
	seed=${seed} \
	temperature=${temperature}
