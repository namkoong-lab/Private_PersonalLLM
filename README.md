# PersonalLLM

**PersonalLLM** is designed to facilitate research in Large Language Model (LLM) personalization. Unlike traditional preference-style datasets, PersonalLLM offers a diverse set of prompts and responses that reflect a wide range of user preferences. Additionally, we provide an evaluation set for various personalization algorithms.

## Getting Started

### Environment Setup

To set up the environment, you can use either `poetry` (preferred) or `conda` with the provided `env.yml` file or `requirements.txt`:

For poetry, run:
```
poetry install
poetry shell
```

For conda, run:
`conda create --name personalllm python=3.10`
`conda env update --file env.yml --prune`

### Dataset Generation

For detailed instructions on generating the dataset, please see the [Dataset Generation Guide](PersonalLLM/data/README.md).

### Dataset Samples

- [PersonalLLM](https://huggingface.co/datasets/as6154/PersonalLLM)
- [PersonalLLM-Eval](https://huggingface.co/datasets/namkoong-lab/PersonalLLM_Eval)

### Personalization Algorithm Evaluation

For comprehensive guidelines on evaluating personalization algorithms, refer to the [Evaluation Guide](PersonalLLM/eval/README.md).


## Paper

The paper can be found [here](https://arxiv.org/abs/2405.10273).

All plots were generated using the code in [paper/visualize.ipynb](paper/visualize.ipynb) file.