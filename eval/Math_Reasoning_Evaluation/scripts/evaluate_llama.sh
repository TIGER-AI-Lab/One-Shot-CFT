set -ex

PROMPT_TYPE="llama"
MODEL_NAME_OR_PATH=$1
OUTPUT_DIR=$2
SUMMARY_PATH=$3
SPLIT="test"
NUM_TEST_SAMPLE=-1

mkdir -p $OUTPUT_DIR
cd ..

# DATA_NAME="aime25,aime24,math-500,minerva_math,olympiadbench,amc23,theoremqa,gsm8k"
# DATA_NAME="minerva_math"
DATA_NAME="aime25,aime24,math-500,minerva_math,amc23,olympiadbench"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --summary_path ${SUMMARY_PATH} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --max_tokens_per_call 4096 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    # --overwrite \


#DATA_NAME="aime24"
#TOKENIZERS_PARALLELISM=false \
#python3 -u math_eval.py \
#    --model_name_or_path ${MODEL_NAME_OR_PATH} \
#    --data_name ${DATA_NAME} \
#    --output_dir ${OUTPUT_DIR} \
#    --summary_path ${SUMMARY_PATH} \
#    --split ${SPLIT} \
#    --prompt_type ${PROMPT_TYPE} \
#    --num_test_sample ${NUM_TEST_SAMPLE} \
#    --seed 0 \
#    --temperature 0.8 \
#    --n_sampling 1 \
#    --top_p 1 \
#    --start 0 \
#    --end -1 \
#    --use_vllm \
#    --save_outputs \
#    # --overwrite \
