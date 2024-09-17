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

class SelfKShotICLAlgorithm(BaseAlgorithm):
    """
    This does not utilize the meta_learning_database, and generates from vllm using train dataset only.

    """

    def __init__(self, args):
        super().__init__(args)
        self.num_shots = args.num_shots
        self.args = args
        
        print("Loading dataset...")
        start_time = time.time()
        self.eval_dataset = load_dataset("andrewsiah/Eval_Pref_Dataset_with_stella_400M_v5_embeddings", cache_dir=args.cache_dir)['test']
        print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
        
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.max_history_tokens = 7000

    def find_winning_losing_responses(self, row: dict, prompt_num: int) -> tuple:
        chosen = row[f'chosen_{prompt_num}']
        winning_response = row[f'response_{prompt_num}_a'] if chosen == 'a' else row[f'response_{prompt_num}_b']
        losing_response = row[f'response_{prompt_num}_b'] if chosen == 'a' else row[f'response_{prompt_num}_a']
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
        
        total_tokens = 0
        for i in range(1, int(row['user_history_length']) + 1):
            past_prompt = row[f'prompt_{i}']
            winning_response, losing_response = self.find_winning_losing_responses(row, i)
            
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
                print(f"WARNING: Dropped {int(row['user_history_length']) - len(history)}/{self.num_shots} shot ICL examples")
                break
            
            history.append(example)
            total_tokens += example_tokens
            
            if len(history) >= self.num_shots:
                break
        
        prompt += "".join(history)
        if self.args.responses_to_include == 'winning_only':
            prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should be similar to the liked responses. "
        elif self.args.responses_to_include == 'losing_only':
            prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should be dissimilar to the disliked responses. "
        elif self.args.responses_to_include == 'losing_as_winning_only':
            prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should be similar to the liked responses. "
        elif self.args.responses_to_include == 'winning_and_losing':
            prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should be similar to the liked responses and dissimilar from the disliked responses. "
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
    test_gen = SelfKShotICLAlgorithm()
    updated_dataset = test_gen.generate_evaluation_responses(debug=True)
    print(updated_dataset[0]["test_response"])