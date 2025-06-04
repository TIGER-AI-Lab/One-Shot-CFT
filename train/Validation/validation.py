import os
import shutil


def main(summary_path, model_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    curr_accu = 60.0
    curr_ckpt = None
    with open(summary_path, 'r') as f:
        for line in f.readlines():
            segs = line.strip().split(" Final Accuracy: ")
            accu = float(segs[1])
            ckpt_num = segs[0].split("-checkpoint-")[1].split("/")[0]
            if accu > curr_accu:
                curr_accu = accu
                curr_ckpt = ckpt_num
    if not curr_ckpt:
        print("Validation error")
        return
    source_folder = os.path.join(model_dir, f"checkpoint-{str(curr_ckpt)}")
    shutil.copytree(source_folder, output_dir)


if __name__ == "__main__":
    main(
        "",
        "",
        ""
    )


