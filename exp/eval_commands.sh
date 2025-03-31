# 8xh100
# m1
make -f exp/250319-ablation/h100.makefile seed=42 temperature=0.0 thinking_bugdet

make -f exp/250319-ablation/h100.makefile seed=42 temperature=0.0 thinking_budget_X-keep_think_below_budget

# baseline
make -f exp/250318-eval-baseline_llm/h100.makefile seed=42 temperature=0.0

# medical
make -f exp/250318-eval-medical_llm/above_10b.makefile seed=42 temperature=0.0

# m1 extrapolate thinking budget
make -f exp/250319-ablation/h100.extraplate_thinking_budget.makefile seed=42 temperature=0.0

# m1 force think
make -f exp/250319-ablation/h100.makefile seed=42 temperature=0.0 thinking_budget_X-keep_think_below_budget thinking_budget_for_keep_think=4096
make -f exp/250319-ablation/h100.makefile seed=42 temperature=0.0 thinking_budget_X-keep_think_below_budget thinking_budget_for_keep_think=8192

# m1 data ablation
make -f exp/250319-ablation/h100.data_curation.makefile seed=42 temperature=0.0


# 8xl40s
# medical
make -f exp/250318-eval-medical_llm/below_10b.makefile seed=42 temperature=0.0
