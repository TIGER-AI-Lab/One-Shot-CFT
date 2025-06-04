# One-Shot-CFT

This repo contains the code for [Unleashing the Reasoning Potential of Pre-trained LLMs by Critique Fine-Tuning on One Problem](https://arxiv.org/abs/). One-Shot Critique Fine-Tuning (CFT) is a simple, robust, and compute-efficient training paradigm for unleashing the reasoning capabilities of pretrained LLMs in both mathematical and logical domains. By leveraging critiques on just one problem, One-Shot CFT enables models like Qwen and LLaMA to match or even outperform reinforcement learning, while using 20× less compute.


<a target="_blank" href="https://github.com/TIGER-AI-Lab/One-Shot-CFT">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-black?style=flat&logo=github"></a>
<a target="_blank" href="https://arxiv.org/abs/">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-green?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://tiger-ai-lab.github.io/One-Shot-CFT">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/datasets/TIGER-Lab/One-Shot-CFT-Data">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/collections/TIGER-Lab/one-shot-cft-683fbb4d2bcf698dbea8fb21">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>
<br>

## Highlights

- **Unleashes Reasoning with One Example:** One-Shot CFT uses critiques of diverse model-generated solutions to a single problem to significantly boost performance across math and logic tasks. For example, with just 5 GPU hours of training on Qwen2.5-Math-7B, One-Shot CFT achieves an average improvement of +15% on six math benchmarks and +16% on three logic reasoning benchmarks.
- **Outperforms RLVR and Full SFT with 20× Less Compute:** One-Shot CFT outperforms both one-shot Reinforcement Learning with Verifiable Rewards (RLVR) and full-dataset supervised fine-tuning, while requiring only 5 GPU hours on a 7B model—offering a much more efficient and stable training alternative.
- **Robust Across Seeds and Model Scales:** One-Shot CFT remains effective across different seed problem choices and model sizes—from 1.5B to 14B parameters—demonstrating strong generalization and scalability.


## Getting Started

### Installation

```bash
cd tools/
bash setup_env.sh
```

### Preparing Datasets

```bash
bash prepare_data.sh
```

### Training

1. Train on Mathematical Reasoning
```bash
cd ../train/
bash train_on_math_reasoning.sh
```

We randomly select 500 math problems (excluding MATH-500) for validation.
To validate after training:
```bash
cd train/Validation
bash start_validate.sh
```
This generates validation_summary.txt containing MATH-Validation scores per checkpoint. Select the checkpoint with the highest score as your final model.


2. Train on Logic Reasoning
```bash
cd ../train/
bash train_on_logic_reasoning.sh
```
We do not use a separate validation set for logic tasks. Based on our experiments, checkpoints between ckpt-30 and ckpt-40 generally yield the best performance.

## Evaluation

Edit the following scripts with your trained model path and output directory:
* eval/eval_on_math_reasoning.sh
* eval/eval_on_logic_reasoning.sh

Then run:

```bash
cd eval/
bash eval_on_math_reasoning.sh
bash eval_on_logic_reasoning.sh
```

Our evaluation code is based on [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) and [BBEH](https://github.com/google-deepmind/bbeh).

## Create Your Own Critique Data

You can create new critique data using the prompt templates in "prompts/" for:
* Candidate solution generation
* Teacher critique generation


## Citation

Cite our paper as
```
@article{wang2025critique,
  title={Critique fine-tuning: Learning to critique is more effective than learning to imitate},
  author={Wang, Yubo and Yue, Xiang and Chen, Wenhu},
  journal={arXiv preprint arXiv:2501.17703},
  year={2025}
}
```


