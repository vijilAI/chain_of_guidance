# ConsistencyTuning
**Improving Consistency in Large Language Models through Chain of Guidance**

ConsistencyTuning is a framework for enhancing the consistency of large language models by leveraging our proposed **Chain of Guidance** prompting technique and fine-tuning methodologies. This approach is designed to benchmark and improve the consistency of LLMs.

---

## Setup

Follow the steps below to set up the environment and dependencies:

### 1. Clone the Repository
```bash
# Clone the repository
git clone https://github.com/vijilAI/altus.git
cd altus

# Initialize axolotl submodule
git submodule init
git submodule update
```

### 2. Create a Virtual Environment
```bash
# Create and activate a virtual environment
conda create -n venv python=3.11 -y
conda activate venv
```

### 3. Install Dependencies
```bash
pip install requirements.txt

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

Explore our **Chain of Guidance** prompting technique to improve the consistency of language models. Detailed examples comparing the consistency scores achieved through our prompting technique versus standard prompting can be found in the following notebook:

- **Notebook**: `altus/consistencybench/notebooks/score_consistency.ipynb`

The data collected using this technique is utilized for fine-tuning models to enhance their consistency. An example fine-tuning dataset, created using the TruthfulQA dataset, can be found at:

- `altus/data/consistency_finetune-data-v1.jsonl`

---

## Fine-tuning

We leverage the [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) library for fine-tuning. A wrapper is built around Axolotl to simplify the fine-tuning process. Follow the notebook below for detailed examples or use the code snippet provided.

- **Notebook**: `notebooks/finetune_axolotl-train.ipynb`

### Fine-tuning Workflow
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