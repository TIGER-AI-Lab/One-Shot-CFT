import os
import json
import argparse
import time
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from tqdm import tqdm
from utils import *
from evaluate import evaluate_correctness


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on BigBench Extra Hard benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output_dir_path", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--summary_path", type=str, required=True, help="Path to save the evaluation summary")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum number of tokens to generate")
    parser.add_argument("--sub_task_list", type=str, default=None, help="List of subtasks to evaluate")
    parser.add_argument("--prompt_template", default="qwen2-5", type=str)
    parser.add_argument("--enable_result", default=True, type=str)

    args = parser.parse_args()
    os.makedirs(args.output_dir_path, exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_path), exist_ok=True)

    llm, sampling_params = load_vllm_model(args)
    tasks = load_tasks(args.sub_task_list)

    task_map = run_vllm(llm, sampling_params, tasks, args.prompt_template, args.output_dir_path, args)
    score_sta = compute_score(task_map, args.output_dir_path)
    statistic(score_sta, args)


def compute_score(task_map, output_dir_path):
    score_sta = {}
    for k, v in task_map.items():
        score_sta[k] = {"right": 0.0, "wrong": 0.0}
        output_res_path = os.path.join(output_dir_path, k + "_eval_result.json")
        for i, each in enumerate(v):
            v[i]["pred"] = ""
            model_output = each["output"]
            gt = each["gt"]
            box_ans = extract_boxed_answer(model_output)
            if box_ans is not None:
                task_map[k][i]["pred"] = box_ans
                model_output = "The answer is: " + box_ans
            score = evaluate_correctness(model_output, gt)
            if score is True:
                score_sta[k]["right"] += 1
            else:
                score_sta[k]["wrong"] += 1
        with open(output_res_path, "w") as f:
            f.write(json.dumps(v, indent=4))
    return score_sta


def compute_harmonic(score_sta):
    accuracies = []
    for k, v in score_sta.items():
        if "bbeh_zz_mini" == k:
            continue
        accuracies.append(v["accu"])
    n = len(accuracies)
    reciprocal_sum = sum(1 / (acc + 0.01) for acc in accuracies)
    return n / reciprocal_sum


def statistic(score_sta, args):
    model_name = "/".join(args.model_path.split("/")[-2:])
    from datetime import datetime
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    total_right = 0.0
    total_wrong = 0.0
    for k, v in score_sta.items():
        if "bbeh_zz_mini" == k:
            continue
        total_right += v["right"]
        total_wrong += v["wrong"]
        score_sta[k]["accu"] = v["right"] / (v["right"] + v["wrong"])
    harmonic_mean = compute_harmonic(score_sta)
    score_sta["aa_total"] = {
        "right": total_right,
        "wrong": total_wrong,
        "micro_average_accu": total_right/ (total_wrong + total_right),
        "harmonic_mean": harmonic_mean}
    if "bbeh_zz_mini" in score_sta:
        score_sta["aa_mini"] = {
            "right": score_sta["bbeh_zz_mini"]["right"],
            "wrong": score_sta["bbeh_zz_mini"]["wrong"],
            "micro_average_accu": score_sta["bbeh_zz_mini"]["right"] / (score_sta["bbeh_zz_mini"]["right"] + score_sta["bbeh_zz_mini"]["wrong"])
        }
    sta_output_path = os.path.join(args.output_dir_path, "sta_result.json")
    with open(sta_output_path, "w") as f:
        f.write(json.dumps(score_sta, indent=4))
    with open(args.summary_path, "a") as f:
        f.write(model_name + "\t" + formatted_time + "\n")
        f.write(json.dumps(score_sta) + "\n")
    print("json.dumps(score_sta)", json.dumps(score_sta))


if __name__ == "__main__":
    main()



