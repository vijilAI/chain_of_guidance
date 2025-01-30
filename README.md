# Chain of Guidance

**Chain of Guidance (CoG)** is a framework for enhancing the consistency of Large Language Models by leveraging a combination of synthetic data generation and fine-tuning (PEFT or SFT).

This paper contains code for the CoG method, introduced in the paper
> *Improving Consistency in Large Language Models through Chain of Guidance*. Harsh Raj, Vipul Gupta, Domenic Rosati, Subhabrata Majumdar, **2025**.

---

## Setup

Follow the steps below to set up the environment and dependencies:

### 1. Clone the Repository
```bash
# Clone the repository
git clone https://github.com/vijilAI/chain_of_guidance.git

# Initialize axolotl submodule
git submodule update --init --recursive
```

### 2. Create a Virtual Environment
```bash
# Create and activate a virtual environment
conda create -n venv python=3.11 -y
conda activate venv
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt

# Install PyTorch with CUDA support
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Navigate to the axolotl submodule
cd finetune/axolotl

# Install additional dependencies
pip install packaging ninja
pip install -e '.[flash-attn,deepspeed]'
```

---

## Benchmarking Consistency

The CoG prompting technique can improve the consistency of LLM outputs. For examples comparing the consistency scores achieved through CoG vis-a-vis standard prompting, check out [this notebook](notebooks/score_consistency.ipynb).

## Fine-tuning

Data generated through applying CoG on a capable LLM (e.g. GPT-4) can be is utilized for fine-tune a smaller LLM to enhance their consistency. [Here](data/consistency_finetune-data-v1.jsonl) is an example dataset, created using CoG-generated paraphrases the TruthfulQA benchmark.

For fine-tuning, we utilize the [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) library, with some wrapper function for easy usage. [This notebook](notebooks/finetune_axolotl_train.ipynb) gives detailed instruction for that purpose. Alternatively, you can use the code snippet below.

```python
from finetune import Finetune, FinetuneConfig

# Define configurations using a YAML file or a Python dictionary
# Example YAML files can be found in './finetune/examples/'
config = FinetuneConfig(<path_to_yaml_file_or_dict>)

# Create a fine-tune object and run the process
finetune = Finetune(config)
finetune.run()

# Retrieve a unique identifier for tracking the fine-tuning process

# Check the status of a specific fine-tuning job
finetune.status(job_id=<id>)
```

## Comparison

Finally, you can use [this notebook](notebooks/base_vs_finetuned_model.ipynb) to compare semantic consistency metrics of the base and CoG-finetuned versions of an LLM.