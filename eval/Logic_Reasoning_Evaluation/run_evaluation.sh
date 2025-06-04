set -ex

MODEL_NAME_OR_PATH=$1
OUTPUT_DIR=$2
SUMMARY_PATH=$3
TASK_LIST="bbeh_causal_understanding,bbeh_disambiguation_qa,bbeh_time_arithmetic"

python3 -u run_evaluate.py \
    --model_path ${MODEL_NAME_OR_PATH} \
    --output_dir_path ${OUTPUT_DIR} \
    --summary_path ${SUMMARY_PATH} \
    --sub_task_list ${TASK_LIST}




