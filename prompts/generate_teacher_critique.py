

def get_prompt_math(question, solution):
    question = "Question:\n" + question
    solution = "Student's Solution:\n" + solution
    prompt = f"You are a mathematics expert. Analyze if the student's solution to the given question is correct. " \
             f"Follow these steps:\n" \
             f"1. Identify the key mathematical concepts and correct approach.\n" \
             f"2. Check each step of the student's solution.\n" \
             f"3. If incorrect, point out errors and provide the correct solution, " \
             f"putting your final answer within \\boxed{{}}.\n" \
             f"Conclude with \"Conclusion: right/wrong [END]\"\n\n{question}\n\n{solution}"
    return prompt


def get_prompt_logic(question, solution):
    question = "Question:\n" + question
    solution = "Student's Solution:\n" + solution
    prompt = f"Analyze if the student's solution to the given question is correct. " \
             f"Follow these steps:\n" \
             f"1. Identify the key mathematical concepts and correct approach.\n" \
             f"2. Check each step of the student's solution.\n" \
             f"3. If incorrect, point out errors and provide the correct solution, " \
             f"putting your final answer within \\boxed{{}}.\n" \
             f"Conclude with \"Conclusion: right/wrong [END]\"\n\n{question}\n\n{solution}"
    return prompt


