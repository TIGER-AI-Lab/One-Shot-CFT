set -x

model_path="../../models/Qwen2.5-Math-7B" # the path to your checkpoints
output_path="../eval_math_results/"
summary_path="../eval_math_results/summary.txt"

cd Math_Reasoning_Evaluation/scripts
bash evaluate_qwen.sh $model_path $output_path $summary_path








