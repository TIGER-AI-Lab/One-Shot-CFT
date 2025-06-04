set -x

model_path="../../models/Qwen2.5-Math-7B" # the path to your checkpoints
output_path="../eval_logic_results/"
summary_path="../eval_logic_results/summary.txt"

cd Logic_Reasoning_Evaluation/
bash run_evaluation.sh $model_path $output_path $summary_path

