# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true


############################################### gsm8k evaluations ###############################################
task=gsm8k
length=256
block_length=32
num_fewshot=0
steps=256

# baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="gsm8k"

# dParallel
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='Zigeng/dParallel-LLaDA-8b-instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="gsm8k"




############################################### minerva_math evaluations ###############################################
task=minerva_math
length=256
block_length=32
num_fewshot=4
steps=256

# baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="minerva_math"

# dParallel
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='Zigeng/dParallel-LLaDA-8b-instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="minerva_math"



############################################### humaneval evaluations ###############################################
task=humaneval
length=256
block_length=32
num_fewshot=0
steps=256

# baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="humaneval" \
--output_path evals_results/baseline/humaneval-ns0-${length} --log_samples

# dparallel
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='Zigeng/dParallel-LLaDA-8b-instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.5,show_speed=True,task="humaneval" \
--output_path evals_results/parallel/humaneval-ns0-${length} --log_samples

## NOTICE: use postprocess for humaneval
python postprocess_code_humaneval.py {the samples_xxx.jsonl file under output_path}





############################################### mbpp evaluations ###############################################
task=mbpp
length=256
block_length=32
num_fewshot=3
steps=256

# baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="mbpp" \
--output_path evals_results/baseline/mbpp-ns0-${length} --log_samples

# parallel
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='Zigeng/dParallel-LLaDA-8b-instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.45,show_speed=True,task="mbpp" \
--output_path evals_results/parallel/mbpp-ns0-${length} --log_samples

## NOTICE: use postprocess for mbpp
python postprocess_code_mbpp.py {the samples_xxx.jsonl file under output_path}

