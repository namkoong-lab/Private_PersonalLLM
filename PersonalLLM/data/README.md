## Files:

1. **gen_responses.py**: This script generates responses from a given prompt set sourced from Hugging Face (for example, see [PersonalLLM_prompts](https://huggingface.co/datasets/namkoong-lab/PersonalLLM_prompts)). The generated responses are then uploaded to Hugging Face. Configuration details can be found in `gen_responses_conf.yaml`.
You need to duplicate the `.env.example` with a `.env` file and add your openrouter key to generate responses.

2. **gen_rewards.py**: This script generates rewards for each (prompt, response) pair using ONE reward model. The generated rewards are subsequently uploaded to Hugging Face. Configuration details can be found in `gen_rewards_conf.yaml`.

3. **gen_all_rewards.py**: This script generates rewards for each (prompt, response) pair using all the reward models from PersonalLLM. The generated rewards are subsequently uploaded to Hugging Face. Configuration details can be found in `gen_all_rewards_conf.yaml`.
`python gen_all_rewards.py --conf_filepath path/to/config.yaml --dataset_name your_dataset_name --output_dataset_name your_output_dataset_name`

![Figure](../assets/metalearn_fig.png)

4. **gen_pref_interaction_database.py** and **gen_reward_interaction_database.py**: These scripts are used to generate conversation histories of varying lengths, as detailed in our paper. These histories are used to create the metalearning database and the evaluation dataset. Examples of these databases can be found at the following links:

    - [MetaLearningPrefDatabase10K](https://huggingface.co/datasets/andrewsiah/MetaLearningPrefDatabase10K)
    - [MetaLearningPrefDatabase1K](https://huggingface.co/datasets/andrewsiah/MetaLearningPrefDatabase1K)

    Evaluation datasets can be found at:

    - [Eval_Pref_Dataset](https://huggingface.co/datasets/namkoong-lab/PersonalLLM_Eval)

5. **gen_personas.ipynb**: This script generates a Dirichlet weighted ensemble of personalities using multiple reward models. It's worth noting that both `gen_pref_interaction_database` and `gen_reward_interaction_database` inherently include this functionality.

6. **meta_learning.py**: This script generates a meta-learning database of previous user interactions for a given dataset. Refer to Section 4.2 of the paper for more details. 