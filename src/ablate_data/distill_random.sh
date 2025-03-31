# decon eval
python src/distill_data/decontaminate_eval.py \
--repo_id mmqm/m196k-random_1k \
--eval_json_list misc/m1_eval_data.json
# No contamination for random_1k

python src/distill_data/decontaminate_eval.py \
--repo_id mmqm/m196k-random_23k \
--eval_json_list misc/m1_eval_data.json


# cost
# input 100; output 700: (100 * 4 + 700 * 16) * 1e-6 = 0.0116

# 1.1k: 1250 * 0.0116 = 14.5
python src/distill_data/distill_data.py \
--repo_id mmqm/m196k-random_1k-decon_eval \
--model_name r1 

# input 100; output 700: (100 * 4 + 1200 * 16) * 1e-6 = 0.0196
# 25k: 30,000 * 0.0196 = 588
python src/distill_data/distill_data.py \
--repo_id mmqm/m196k-random_23k-decon_eval \
--model_name r1



# random 23k: mmqm/m196k-random_23k-decon_eval-r1-filter_wrong
# random 1k: mmqm/m196k-random_1k-decon_eval-r1-filter_wrong
# hard random 1k: mmqm/m196k-dedup-decon-filter_easy-r1-filter_wrong-decon_eval-hard_random_1k
# hard-domain random 1k: mmqm/m196k-dedup-decon-filter_easy-r1-filter_wrong-decon_eval-domain_1k


python src/tokenize_data/apply_chat_template.py --repo_id mmqm/m196k-random_23k-decon_eval-r1-filter_wrong
python src/tokenize_data/apply_chat_template.py --repo_id mmqm/m196k-random_1k-decon_eval-r1-filter_wrong
python src/tokenize_data/apply_chat_template.py --repo_id mmqm/m196k-dedup-decon-filter_easy-r1-filter_wrong-decon_eval-hard_random_1k
python src/tokenize_data/apply_chat_template.py --repo_id mmqm/m196k-dedup-decon-filter_easy-r1-filter_wrong-decon_eval-domain_1k