import json
import os


def main():
    with open("test.json", "r") as fi:
        data = json.load(fi)
    output_data = []
    for each in data:
        question = each["Question"]
        answer = str(each["Answer"])
        if answer == "True" or answer == "False":
            question += " Answer with \\boxed{True} or \\boxed{False}."
        output_data.append({"problem": question, "answer": answer})
    with open("test.jsonl", "w") as fo:
        for each in output_data:
            fo.write(json.dumps(each) + "\n")


main()


