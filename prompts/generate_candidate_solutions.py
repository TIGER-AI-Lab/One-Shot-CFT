

def get_prompt(question, model_type):
    question = "Question:\n" + question
    if model_type == "qwen3":
        prompt = f"<|im_start|>user\nPlease reason step by step to find a solution to the following " \
             f"question, and put your final answer within \\boxed{{}}.\n{question}<|im_end|>\n" \
             f"<|im_start|>assistant\n"
    elif model_type == "qwen2-5":
        prompt = f"<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n" \
                 f"<|im_start|>user\n{question}<|im_end|>\n" \
                 f"<|im_start|>assistant\n"
    elif model_type == "phi4":
        prompt = f"<|im_start|>system<|im_sep|>\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n" \
                 f"<|im_start|>user<|im_sep|>\n{question}<|im_end|>\n" \
                 f"<|im_start|>assistant<|im_sep|>\n"
    else:
        print("unsupported model type")
        return None
    return prompt


