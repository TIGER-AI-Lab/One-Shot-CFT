import json
from vllm import LLM, SamplingParams
import os
from typing import List, Dict, Any, Optional
from evaluate import evaluate_correctness

def load_vllm_model(args):
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        if "max_position_embeddings" in config and config["max_position_embeddings"] < 8192:
            config["max_position_embeddings"] = 8192
            with open(config_path, "w") as fo:
                json.dump(config, fo, indent=4)

    generation_config_path = os.path.join(args.model_path, "generation_config.json")
    with open(generation_config_path, "r") as f:
        config = json.load(f)
        if "max_new_tokens" in config and config["max_new_tokens"] < 4096:
            config["max_new_tokens"] = 4096
            with open(generation_config_path, "w") as fo:
                json.dump(config, fo, indent=4)
    try:
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
        llm = LLM(
            model=args.model_path,
            trust_remote_code=True,
            tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))  # 根据GPU数量调整
        )

        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n=1,
            stop=stop_words,
            stop_token_ids=(
                [151645, 151643]
                if "qwen2" in args.model_path.lower()
                else None
            ),
            top_p=args.top_p
        )
        return llm, sampling_params
    except Exception as e:
        print("load vllm model failed", e)
        return None, None


def batch_predict(llm, sampling_params, prompts: List[str]) -> List[str]:
    if not llm or not sampling_params:
        print("llm, sampling_params are None")
        return []
    try:
        print("Processing", len(prompts))
        outputs = llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
        return results
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return []


def load_tasks(task_list_str):
    if not task_list_str:
        task_list = None
    else:
        task_list = task_list_str.split(",")
    task_map = {}
    for sub_dir in os.listdir("benchmark_tasks"):
        if sub_dir.startswith("bbeh_"):
            if task_list and sub_dir not in task_list:
                continue
            print(sub_dir)
            data_path = os.path.join(os.path.join("benchmark_tasks", sub_dir, "task.json"))
            with open(data_path, "r") as f:
                data = json.load(f)
                task_map[sub_dir] = data
    return task_map


def build_prompt(prompt_type, question):
    if prompt_type == "qwen2-5":
        template = ("<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n",
                    f"<|im_start|>user\n{question}<|im_end|>\n",
                    "<|im_start|>assistant\n")
        prompt = "\n".join(template)
    elif prompt_type == "qwen3":
        template = ("",
                    f"<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{question}<|im_end|>\n",
                    "<|im_start|>assistant\n<think>\n")
        prompt = "\n".join(template)
    else:
        print("unsupported prompt type", prompt_type)
        template = (
            "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n",
            f"<|im_start|>user\n{question}<|im_end|>\n",
            "<|im_start|>assistant\n")
        prompt = "\n".join(template)
    return prompt


def run_vllm(llm, sampling_params, tasks, prompt_type, output_dir_path, args, enable_resume=True):
    task_map = {}
    score_sta = {}
    for k, v in tasks.items():
        score_sta[k] = {"right": 0.0, "wrong": 0.0}
        output_res_path = os.path.join(output_dir_path, k + "_eval_result.json")
        if os.path.exists(output_res_path) and enable_resume:
            with open(output_res_path, "r") as f:
                results = json.load(f)
            if len(results) == len(v["examples"]):
                task_map[k] = results
                continue
        print("running on sub task", k)
        questions = v["examples"]
        prompts = []
        gt = []
        for each in questions:
            prompts.append(build_prompt(prompt_type, each["input"]))
            gt.append(each["target"])
        print("len(prompts)", len(prompts))
        outputs = batch_predict(llm, sampling_params, prompts)
        if len(outputs) != len(questions):
            print("inconsistent length of the output and questions", len(outputs), len(questions))
        results = []
        for i, each in enumerate(questions):
            model_output = outputs[i]
            box_ans = extract_boxed_answer(model_output)
            pred = None
            if box_ans is not None:
                pred = box_ans
                model_output = "The answer is: " + box_ans
            score = evaluate_correctness(model_output, gt[i])
            results.append({"question": each, "output": outputs[i], "gt": gt[i],
                            "pred": pred, "score": score})
            if score is True:
                score_sta[k]["right"] += 1
            else:
                score_sta[k]["wrong"] += 1
        accu = score_sta[k]["right"] / (score_sta[k]["right"] + score_sta[k]["wrong"])
        print(k, "accu:", accu)
        with open(args.summary_path, "a") as f:
            f.write("/".join(args.model_path.split("/")[-2:]) + "\t" + k + "\t" + str(accu) + "\n")
        with open(output_res_path, "w") as f:
            f.write(json.dumps(results, indent=4))
        task_map[k] = results
    return task_map


def extract_boxed_answer(pred_str: str):
    if "boxed" not in pred_str:
        return None
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return None
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a










