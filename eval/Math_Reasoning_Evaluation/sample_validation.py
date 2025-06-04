import json
import random

def sample_diff_jsonl(path_1, path_2, output_path, sample_size=500):
    with open(path_1, 'r', encoding='utf-8') as f:
        subset_data = [json.loads(line.strip()) for line in f if line.strip()]
    math_500 = {}
    for each in subset_data:
        math_500[each["problem"]] = each

    with open(path_2, 'r', encoding='utf-8') as f:
        full_data = [json.loads(line.strip()) for line in f if line.strip()]

    remaining = []
    for each in full_data:
        if each["problem"] in math_500:
            continue
        remaining.append(each)

    print(len(remaining))

    if len(remaining) < sample_size:
        raise ValueError(f"not enough")

    sampled = random.sample(remaining, sample_size)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for line in sampled:
            out_f.write(json.dumps(line) + '\n')

    print(sample_size)

if __name__ == '__main__':
    sample_diff_jsonl("data/math-500/test.jsonl",
                      "data/math/test.jsonl",
                      "data/math-500-validation/valid.jsonl")


