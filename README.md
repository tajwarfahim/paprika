# Training a Generally Curious Agent
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/metaworld/blob/master/LICENSE)

This is the official PyTorch implementation of our paper ["Training a Generally Curious Agent"](https://arxiv.org/abs/2502.17543) by [Fahim Tajwar*](https://tajwarfahim.github.io/), [Yiding Jiang*](https://yidingjiang.github.io/), [Abitha Thankaraj](https://abitha-thankaraj.github.io/), [Sumaita Sadia Rahman](https://www.linkedin.com/in/sumaitasr/), [J Zico Kolter](https://zicokolter.com/), [Jeff Schneider](https://www.cs.cmu.edu/~schneide/), and [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/). Please see the [project website](https://paprika-llm.github.io/) for more information about this work. For any questions/concerns related to the codebase, please reach out to [Fahim Tajwar](mailto:tajwarfahim932@gmail.com).

## Citation

If you use this repo in your research, please consider citing our paper:

```
@misc{tajwar2025traininggenerallycuriousagent,
      title={Training a Generally Curious Agent}, 
      author={Fahim Tajwar and Yiding Jiang and Abitha Thankaraj and Sumaita Sadia Rahman and J Zico Kolter and Jeff Schneider and Ruslan Salakhutdinov},
      year={2025},
      eprint={2502.17543},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.17543}, 
}
```

## Installation

In order for the installations to go smoothly, make sure you are operating from a GPU machine, typically one compatible with flash attention. It is ideal if you use the same GPU machines that you would use to run your experiments. 

Please create a new conda environment with the correct dependencies (these may differ based on your compute resources, please update the packages accordingly). Make sure you are in the correct directory (one that contains llm_exploration, notebooks and scripts sub-directories), and then run the following commands:

```
conda env create -f environment.yml
conda activate paprika
pip install flash-attn --no-build-isolation
pip install -e .
```

## Running experiments

First, make sure you are in the "scripts/bash_scripts" directory. Next, put the OPENAI API key and Hugging Face authentication token as ENV variables. One can put these keys in the bashrc file for simplicity, or put them in individual bash scripts.

```
# Put your openai API key here
export OAI_KEY="<API_KEY>"
export OPENAI_API_KEY="<API_KEY>"

# Put huggingface authentication token here
export HF_TOKEN="<HUGGINGFACE_AUTHENTICATION_TOKEN>"
```

**Run Evaluations on PAPRIKA tasks:**

```
bash run_evaluation.sh
```

Please change the parameters within "run_evaluation.sh" if you want to change the task group, or the exact set of tasks you want to run evaluations on.


**Run Supervised Finetuning**

Our supervised finetuning dataset can be found in this [link](https://huggingface.co/datasets/ftajwar/paprika_SFT_dataset). Please download the dataset, and update the local paths within the bash script "run_supervised_finetuning.sh", and then run it. Our default code uses 8 L40S GPUs and 400GB of memory, but adjust the script accordingly if you have a different amount of resources.

```
bash run_supervised_finetuning.sh
```

**Precompute Reference Model's Log Probs**

For preference finetuning (DPO or RPO), we typically need to calculate the log probabilities of the trajectories using the reference/starting policy. This requires storing the reference policy in GPU memory throughout training (or store it in CPU memory and put it into GPU while calculating the reference log probabilities and moving it back into CPU afterwards). To simplify this, we choose to pre-calculate the log probabilities of our trajectories using the reference model and save it on disk before running DPO or RPO. This saves both time and memory during training. Our preference finetuning dataset can be found in this [link](https://huggingface.co/datasets/ftajwar/paprika_preference_dataset).

In order to precompute and store reference model's log probs, use the following script (after appropriately modifying the local paths):

```
bash precompute_log_probs.sh
```

This would store the tokenized trajectories (preferred and dispreferred), labels (with user/environment tokens masked out) and reference log probabilites into a ".pt" file specified via the config or command line arguments.

**Run Preference Finetuning**

Finally, once the reference log probabilities are pre-calculated, we can use the following bash script (after appropriately modifying the local paths) to train a model using DPO/RPO (please change the hyperparameters within the script as necessary):

```
bash run_preference_finetuning.sh
```

## Other Details

Please see "llm_exploration/game/game_configs" for all the prompts and individual tasks for each task group, "llm_exploration/game/game.py" for our implementation of multi-turn interactions between an agent and a task environment (plus optionally a task judge), "llm_exploration/inference" for how the logic of all task groups and inference using LLMs are implemented, and "llm_exploration/llm_finetuning" for our implementation of multi-turn SFT and DPO optimization routines. It is easy to add another training algorithm/task group by simply adding the desired files in these directories.

## Acknowledgements

The codebase for the algorithm is built on top of the [Hugging Face Trainer implementation](https://huggingface.co/docs/transformers/en/main_classes/trainer) and other functionalities provided by the [Hugging Face Transformers library](https://github.com/huggingface/transformers). We thank the authors of these repositories for providing us with easy-to-work-with codebases.