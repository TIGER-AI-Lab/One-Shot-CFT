import json
from datasets import load_dataset
import os


def download_aime25_dataset(output_dir='../data/aime25/', filename='test.jsonl'):
    try:
        dataset = load_dataset("math-ai/aime25")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 构建完整的输出文件路径
        output_path = os.path.join(output_dir, filename)

        # 将数据集转换为字典，然后保存为JSON
        dataset_dict = {split: dataset[split].to_dict() for split in dataset.keys()}
        problems = dataset_dict["test"]["problem"]
        answers = dataset_dict["test"]["answer"]
        ids = dataset_dict["test"]["id"]
        output_data = []
        for i in range(len(ids)):
            curr = {"id": ids[i], "problem": problems[i], "answer": answers[i], "question": problems[i]}
            output_data.append(curr)

        with open(output_path, 'w', encoding='utf-8') as f:
            for each in output_data:
                f.write(json.dumps(each) + "\n")
        print(f"数据集已成功保存至: {output_path}")
        return output_path

    except Exception as e:
        print(f"下载或保存数据集时出错: {str(e)}")
        raise


if __name__ == "__main__":
    # 调用函数测试
    download_aime25_dataset()

