This repository contains a variety of programs used in the paper **Stop Wasting Time Fine-Tuning: Traditional Classifiers Shine with LLM Embeddings for Political Textual Analysis**. In this work, we use 19 large language models (LLMs) to demonstrate that using LLMs to create text embeddings for downstream classifiers provides performance comparable to or exceeding fine-tuning LLMs for classification. We also demonstrate that the embed-then-classify pipeline is significantly faster and less data intensive than a finetune-then-classify pipeline. 

# Setup
### Step 1: Set Up a Conda Environment
It is recommended to use Python 3.12.1 for this project to ensure package compatability. Otherwise, additional effort will need to be done to resolve dependency issues. To set and activate up a conda environment, run the following commands:

```bash
conda create -n stop-fine-tuning python=3.12.1
conda activate stop-fine-tuning
```

### Step 2: Run the Setup Script
After setting up the conda environment and installing the necessary dependencies, navigate to the root directory of this repository and run the following command:

```bash
pip install -v -e .
```
This command will install all the required packages listed in the requirements.txt file.

# Usage
Each experiment can be replicated using the provided Python scripts. Note that our analysis uses models of varied sizes. Some LLMs may exceed the compute available to users. In those cases, either substitute the offending models with smaller models or comment out the applications that exceed the available compute resources. All trials presented on the paper were conducted using an NVIDIA H100 Tensor Core GPU, and rerunning these experiments without a GPU will take significantly longer. 

# Data
The data for this project are accessible on our Zenodo repository. We provide raw text files, the entire labeled dataset, and a train/validation/test split used for the experiments. A link to the data is forthcoming.  

# Paper and Citation
If you use this code, please use the following citation: 

```
@article{Coil2025Finetuning,
  title={Stop Wasting Time Fine-Tuning: Traditional Classifiers Shine with LLM Embeddings for Political Textual Analysis},
  author={Coil, Collin},
  journal={Unpublished manuscript},
  year={2025}
}
```
A link to the paper will be published here once it is available. 
