# Quickstart

## Execution Guide:

1. Change directory to 'eval': `cd eval`
2. Activate the 'personal' conda environment: `conda activate personal`

Run the evaluation script with different algorithms:
- TestAlgorithm in Debug Mode: `python eval.py --algorithm=TestAlgorithm --debug`
- KShotICLAlgorithm in Debug Mode: ```python eval.py --algorithm=KShotICLAlgorithm --eval_type=pairwise_pref --num_shots=3 --responses_to_include=winning_and_losing --debug```
- To execute the sweep script:
`python sweep.py`


# Developing New Personalization Algorithms

## Overview
This repository contains various algorithms for generating personalized responses using in-context learning techniques. Each algorithm leverages different strategies to construct prompts and generate responses based on user interactions and preferences.

## Developing a New Algorithm
To create a new algorithm for PersonalLLM, follow these steps:

1. Create a new Python file in the `PersonalLLM/eval/algorithms/` directory, naming it appropriately (e.g., `YourNewAlgorithm.py`).

2. Import the necessary modules and the `BaseAlgorithm` class:

   ```python
   from algorithms.BaseAlgorithm import BaseAlgorithm
   from datasets import Dataset
   import logging
   ```

3. Define your algorithm class, inheriting from `BaseAlgorithm`:

   ```python
   class YourNewAlgorithm(BaseAlgorithm):
       def __init__(self, args):
           super().__init__(args)
           self.logger = logging.getLogger(__name__)
           # Initialize any additional attributes here
   ```

4. Implement the required methods:
   - `generate_pairwise_pref_prompt(self, row: dict) -> str`: This method should generate a prompt based on the input row.
   - `generate_responses(self, prompts: list)`: This method should generate responses for the given prompts.
   - `generate_evaluation_responses(self, args) -> Dataset`: This method should generate evaluation responses for the entire dataset.

5. Reference existing algorithms for implementation details.

6. To use your new algorithm, update the main evaluation script to import and instantiate your algorithm class. Then run 
`python eval.py --algorithm=YourNewAlgorithm --eval_type=pairwise_pref`


## Algorithm Examples
Refer to our paper for explanation for the implementation of the following algorithms.

### MetaLearnKShotICLAlgorithm
This algorithm is the implementation of Section 4.2 in the paper. It finds similar users and prompts from an interaction history dataset, then constructs in-context learning examples for generation.

### LookUpMetaLearnKShotICLAlgorithm
This algorithm is an optimized version of MetaLearnKShotICLAlgorithm, using pre-computed data for faster lookup. It finds users with similar prompts, ranks them, and uses their data to construct in-context learning examples for generation

### KShotICLAlgorithm
This algorithm implements a basic k-shot in-context learning approach. It finds similar prompts from a dataset, constructs in-context learning examples, and generates a response.

### SelfKShotICLAlgorithm
This algorithm uses the user's own history for in-context learning. It constructs examples from the user's past interactions, including liked and disliked responses to generate a response.