set -x

model_path="../../../models/Qwen2.5-Math-7B" # the path to your checkpoints
output_path="../../eval_math_results/"
summary_path="../../eval_math_results/summary.txt"

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd Math_Reasoning_Evaluation/scripts

mkdir -p output_path
bash evaluate_qwen.sh $model_path $output_path $summary_path








