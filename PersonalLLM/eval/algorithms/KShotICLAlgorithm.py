import random
import string
from algorithms.BaseAlgorithm import BaseAlgorithm
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from tqdm import tqdm
from transformers import AutoTokenizer

class KShotICLAlgorithm(BaseAlgorithm):
    """
    This does not utilize the meta_learning_database, and generates from vllm using train dataset only.

    """

    def __init__(self, args):
        super().__init__(args)
        self.num_shots = args.num_shots
        self.args = args
        
        print("Loading datasets...")
        start_time = time.time()
        self.eval_dataset = load_dataset("andrewsiah/Eval_Pref_Dataset_with_stella_400M_v5_embeddings", cache_dir=args.cache_dir)['test']
        self.train_dataset = load_dataset("andrewsiah/PersonalLLM_with_stella_400M_v5_embeddings", cache_dir=args.cache_dir)['train']
        print(f"Datasets loaded in {time.time() - start_time:.2f} seconds")
        
        # Pre-compute all embeddings
        print("Pre-computing embeddings...")
        start_time = time.time()
        self.all_embeddings = np.array(self.train_dataset['prompt_embedding'])
        print(f"Embeddings pre-computed in {time.time() - start_time:.2f} seconds")

        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.max_history_tokens = 7000

    def generate_response(self, row: dict) -> str:
        prompt = "Below are some examples of the user's past conversation history."
        for i in range(int(row["user_history_length"])):
            past_prompt = row["prompt_" + str(i + 1)]
            past_response = row["response_" + str(i + 1)]
            past_reward = str(row["reward_" + str(i + 1)])
            prompt += (
                "User: "
                + past_prompt
                + "\nAssistant: "
                + past_response
                + "\nReward: "
                + past_reward
                + "\n\n"
            )
        prompt += "Use the contexts above to generate a good response for the user prompt below. Stop after answering the User Prompt, don't give a reward.\n"
        prompt += "User: " + row["test_prompt"]
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=512)
        output = self.llm.generate([prompt], sampling_params)
        return output[0].outputs[0].text

    def generate_evaluation_responses(self, debug=False) -> Dataset:
        dataset = self.eval_dataset
        if debug:
            dataset = dataset.select(range(20))
        dataset = dataset.add_column(
            "test_response",
            [self.generate_response(dataset[i]) for i in range(len(dataset))],
        )
        return dataset

    def find_similar_prompts(self, test_prompt_embedding: list, k: int) -> list:
        similarities = cosine_similarity([test_prompt_embedding], self.all_embeddings)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        similar_prompts = [
            {'prompt': self.train_dataset[int(idx)]['prompt'], 'idx': int(idx)}
            for idx in top_k_indices
        ]
        return similar_prompts

    def find_winning_losing_responses(self, idx: int) -> tuple:
        # Get the person_weight from the eval dataset
        person_weight = np.array(self.eval_dataset[0]['person_weight'])[:10]  # Assuming person_weight is the same for all rows
        
        # Calculate the total reward for each response
        total_rewards = []
        for i in range(1, 9):  # 8 responses per prompt
            rewards = np.array([
                self.train_dataset[idx][f'response_{i}_gemma_2b'],
                self.train_dataset[idx][f'response_{i}_gemma_7b'],
                self.train_dataset[idx][f'response_{i}_mistral_raft'],
                self.train_dataset[idx][f'response_{i}_mistral_ray'],
                self.train_dataset[idx][f'response_{i}_mistral_weqweasdas'],
                self.train_dataset[idx][f'response_{i}_llama3_sfairx'],
                self.train_dataset[idx][f'response_{i}_oasst_deberta_v3'],
                self.train_dataset[idx][f'response_{i}_beaver_7b'],
                self.train_dataset[idx][f'response_{i}_oasst_pythia_7b'],
                self.train_dataset[idx][f'response_{i}_oasst_pythia_1b']
            ])
            total_reward = np.dot(person_weight, rewards)
            total_rewards.append(total_reward)
        
        # Find the winning and losing responses
        winning_idx = np.argmax(total_rewards)
        losing_idx = np.argmin(total_rewards)
        
        winning_response = self.train_dataset[idx][f'response_{winning_idx + 1}']
        losing_response = self.train_dataset[idx][f'response_{losing_idx + 1}']
        
        return winning_response, losing_response

    def generate_pairwise_pref_prompt(self, row: dict) -> str:
        prompt = "Below are some examples of past conversations with "
        if self.args.responses_to_include == 'winning_only':
            prompt += "liked responses per prompt."
        elif self.args.responses_to_include == 'losing_only':
            prompt += "disliked responses per prompt."
        elif self.args.responses_to_include == 'losing_as_winning_only':
            prompt += "liked responses per prompt."
        elif self.args.responses_to_include == 'winning_and_losing':
            prompt += "liked and disliked responses per prompt."
        history = []
        
        test_prompt_embedding = row['test_prompt_embedding']
        similar_prompts = self.find_similar_prompts(test_prompt_embedding, self.num_shots )  # Get more prompts than needed
        
        total_tokens = 0
        for similar_prompt in similar_prompts:
            past_prompt = similar_prompt['prompt']
            winning_response, losing_response = self.find_winning_losing_responses(similar_prompt['idx'])
            
            if self.args.responses_to_include == 'winning_only':
                example = f"User: {past_prompt}\nLiked Response: {winning_response}\n\n"
            elif self.args.responses_to_include == 'losing_only':
                example = f"User: {past_prompt}\nDisliked Response: {losing_response}\n\n"
            elif self.args.responses_to_include == 'losing_as_winning_only':
                example = f"User: {past_prompt}\nLiked Response: {losing_response}\n\n"
            elif self.args.responses_to_include == 'winning_and_losing':
                example = f"User: {past_prompt}\nLiked Response: {winning_response}\nDisliked Response: {losing_response}\n\n"
            
            example_tokens = len(self.tokenizer.encode(example))
            if total_tokens + example_tokens > self.max_history_tokens:
                print(f"WARNING: Dropped {len(similar_prompts) - len(history)}/{self.num_shots} shot ICL examples")
                break
            
            history.append(example)
            total_tokens += example_tokens
            
            if len(history) >= self.num_shots:
                break
        
        prompt += "".join(history)
        if self.args.responses_to_include == 'winning_only':
            prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should be similar to the liked responses. "
        elif self.args.responses_to_include == 'losing_only':
            prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should dissimilar to the disliked responses. "
        elif self.args.responses_to_include == 'losing_as_winning_only':
            prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should be similar to the liked responses. "
        elif self.args.responses_to_include == 'winning_and_losing':
            prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should be similar to the liked responses and different from the disliked responses. "
        prompt += f"User: {row['test_prompt']} \nResponse: "
        return prompt

    def generate_responses(self, prompts: list):
        sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)
        outputs = self.llm.generate(prompts, sampling_params)
        return outputs

    def generate_evaluation_responses(self, args) -> Dataset:
        debug = args.debug
        dataset = self.eval_dataset
        if debug:
            dataset = dataset.select(range(50))

        print("Generating prompts...")
        start_time = time.time()
        prompts = [
            self.generate_pairwise_pref_prompt(dataset[i]) for i in tqdm(range(len(dataset)))
        ]
        print(f"Generated prompts in {time.time() - start_time:.2f} seconds")

        print("Generating responses...")
        start_time = time.time()
        outputs = self.generate_responses(prompts)
        responses = [output.outputs[0].text for output in outputs]
        print(f"Generated responses in {time.time() - start_time:.2f} seconds")

        dataset = dataset.add_column("test_response", responses)
        return dataset


if __name__ == "__main__":
    test_gen = KShotICLAlgorithm()
    updated_dataset = test_gen.generate_evaluation_responses(debug=True)
    print(updated_dataset[0]["test_response"])