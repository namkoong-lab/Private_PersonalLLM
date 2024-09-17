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
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MetaLearnKShotICLAlgorithm(BaseAlgorithm):
    """
    MetaLearnKShotICLAlgorithm is an algorithm that uses Meta-Learning to generate pairwise preference prompts.

    First it loops through the meta-learning database to find similar users to the test user.
    Then it finds similar prompts among the similar users.
    Then it generates a pairwise preference prompt using the similar prompts.
    Then it generates responses to the prompt using the LLM.
    """
    def __init__(self, args):
        super().__init__(args)
        self.num_shots = args.num_shots
        self.args = args
        
        # New parameters
        self.user_embedding_type = args.user_embedding_type
        self.user_embedding_computed_from_n_pr = args.user_embedding_computed_from_n_pr
        self.n_similar_users_to_retrieve = args.n_similar_users_to_retrieve
        
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Loading datasets...")
        start_time = time.time()
        self.eval_dataset = load_dataset("andrewsiah/Eval_Pref_Dataset_with_stella_400M_v5_embeddings", cache_dir=args.cache_dir)['test']
        self.meta_learning_database = load_dataset(args.meta_learning_dataset, cache_dir=args.cache_dir)['train']
        self.logger.info(f"Datasets loaded in {time.time() - start_time:.2f} seconds")
        
        # Pre-compute user embeddings and data dict
        self.logger.info("Pre-computing user embeddings and data...")
        start_time = time.time()
        user_embeddings_dict, self.user_data_dict = self.compute_user_embeddings_and_data_dict()
        self.user_embeddings = np.array(list(user_embeddings_dict.values()))
        self.user_ids = list(user_embeddings_dict.keys())
        self.logger.info(f"User embeddings and data pre-computed in {time.time() - start_time:.2f} seconds")

        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.max_history_tokens = 7000

    def compute_user_embeddings_and_data_dict(self):
        user_embeddings = {}
        user_data_dict = {}
        for user in self.meta_learning_database:
            user_id = user['person_id']
            embedding = user[f'user_embedding_avg_{self.user_embedding_type}_{self.user_embedding_computed_from_n_pr}']
            user_embeddings[user_id] = embedding
            user_data_dict[user_id] = user
        
        self.logger.info(f"Computed embeddings and data for {len(user_embeddings)} users")
        return user_embeddings, user_data_dict

    def find_similar_users(self, test_user_embedding):
        start_time = time.time()
        similarities = cosine_similarity([test_user_embedding], self.user_embeddings)[0]
        top_k_indices = similarities.argsort()[-self.n_similar_users_to_retrieve:][::-1]
        similar_users = [self.user_ids[idx] for idx in top_k_indices]
        self.logger.debug(f"Found {len(similar_users)} similar users in {time.time() - start_time:.4f} seconds")
        return similar_users

    def find_similar_prompts(self, test_prompt_embedding: list, similar_users: list, k: int) -> list:
        start_time = time.time()
        all_prompt_embeddings = []
        all_prompt_info = []

        for user_id in similar_users:
            user_data = self.user_data_dict[user_id]
            for i in range(1, self.user_embedding_computed_from_n_pr):
                prompt_key = f"prompt_{i}"
                prompt_embedding_key = f"prompt_{i}_embedding"
                if prompt_key in user_data and prompt_embedding_key in user_data:
                    all_prompt_embeddings.append(user_data[prompt_embedding_key])
                    all_prompt_info.append(
                        {
                            "prompt": user_data[prompt_key],
                            "user_id": user_id,
                            "prompt_index": i,
                        }
                    )

        all_prompt_embeddings = np.array(all_prompt_embeddings)

        similarities = cosine_similarity([test_prompt_embedding], all_prompt_embeddings)[0]
        similarity_prompt_pairs = list(zip(similarities, all_prompt_info))

        similar_prompts = sorted(similarity_prompt_pairs, key=lambda x: x[0], reverse=True)[:k]

        result = [
            {**prompt_info, "similarity": similarity}
            for similarity, prompt_info in similar_prompts
        ]
        self.logger.debug(f"Found {len(result)} similar prompts in {time.time() - start_time:.4f} seconds")
        return result

    def find_winning_losing_responses(self, user_id: int, prompt_index: int) -> tuple:
        start_time = time.time()
        user_data = self.user_data_dict[user_id]
        winning_response = user_data[f'response_{prompt_index}_a']
        losing_response = user_data[f'response_{prompt_index}_b']
        if user_data[f'chosen_{prompt_index}'] == 'b':
            winning_response, losing_response = losing_response, winning_response
        self.logger.debug(f"Found winning and losing responses for user {user_id}, prompt {prompt_index} in {time.time() - start_time:.4f} seconds")
        return winning_response, losing_response

    def generate_pairwise_pref_prompt(self, row: dict) -> tuple:
        start_time = time.time()
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
        
        # Get embeddings for the test user and prompt
        test_user_embedding = row[f'user_embedding_avg_{self.user_embedding_type}']
        test_prompt_embedding = row['test_prompt_embedding']
        
        # Find similar users
        similar_users = self.find_similar_users(test_user_embedding)
        
        # Find similar prompts among the similar users
        similar_prompts = self.find_similar_prompts(test_prompt_embedding, similar_users, self.num_shots)

        total_tokens = 0
        for similar_prompt in similar_prompts:
            past_prompt = similar_prompt['prompt']
            winning_response, losing_response = self.find_winning_losing_responses(similar_prompt['user_id'], similar_prompt['prompt_index'])
            
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
                self.logger.warning(f"WARNING: Dropped {len(similar_prompts) - len(history)}/{self.num_shots} shot ICL examples")
                break
            
            history.append(example)
            total_tokens += example_tokens
            
            if len(history) >= self.num_shots:
                break
        
        prompt += "".join(history)
        if self.args.responses_to_include == 'winning_only':
            prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should be similar to the winning responses. "
        elif self.args.responses_to_include == 'losing_only':
            prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should be dissimilar from the losing responses. "
        elif self.args.responses_to_include == 'losing_as_winning_only':
            prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should be similar to the winning responses. "
        elif self.args.responses_to_include == 'winning_and_losing':
            prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should be similar to the winning responses and dissimilar from the losing responses. "
        prompt += f"User: {row['test_prompt']} \nResponse: "

        # Log test_prompt and best_response
        self.logger.debug(f"Test prompt: {row.get('test_prompt', 'Not found')}")
        self.logger.debug(f"Best response: {row.get('best_response', 'Not found')}")

        # Log the final prompt
        self.logger.debug("Final prompt passed to the language model:")
        self.logger.debug(prompt)
        
        self.logger.debug(f"Generated pairwise preference prompt in {time.time() - start_time:.4f} seconds")
        return prompt, similar_prompts, prompt

    def generate_responses(self, prompts: list):
        start_time = time.time()
        sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)
        outputs = self.llm.generate(prompts, sampling_params)
        self.logger.info(f"Generated {len(prompts)} responses in {time.time() - start_time:.2f} seconds")
        return outputs

    def generate_evaluation_responses(self, args) -> Dataset:
        debug = args.debug
        dataset = self.eval_dataset
        if debug:
            dataset = dataset.select(range(50))

        self.logger.info(f"Initial dataset size: {len(dataset)}")

        self.logger.info("Generating prompts...")
        start_time = time.time()
        
        import os
        max_workers = min(128, os.cpu_count())
        self.logger.info(f"Using {max_workers} workers, os.cpu_count() = {os.cpu_count()}")
        MAX_RETRIES = 3
        TIMEOUT = 180
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.generate_pairwise_pref_prompt, row) for row in dataset]
            prompts = [None] * len(dataset)
            similar_prompts = [None] * len(dataset)
            final_prompts = [None] * len(dataset)
            
            for i, future in enumerate(tqdm(futures, total=len(futures))):
                retries = 0
                while retries < MAX_RETRIES:
                    try:
                        result = future.result(timeout=TIMEOUT)
                        prompt, similar_prompt, final_prompt = result
                        prompts[i] = prompt
                        similar_prompts[i] = similar_prompt
                        final_prompts[i] = final_prompt
                        break
                    except TimeoutError:
                        retries += 1
                        self.logger.warning(f"Prompt Generation for index {i} timed out. Retry {retries}/{MAX_RETRIES}")
                        if retries < MAX_RETRIES:
                            future = executor.submit(self.generate_pairwise_pref_prompt, dataset[i])
                        else:
                            self.logger.error(f"Prompt Generation for index {i} failed after {MAX_RETRIES} retries")
                    except Exception as exc:
                        self.logger.error(f"Prompt Generation for index {i} generated an exception: {exc}")
                        break
    
        # Remove any None values from prompts list
        valid_prompts = [p for p in prompts if p is not None]
        self.logger.info(f"Generated {len(valid_prompts)} prompts in {time.time() - start_time:.2f} seconds")

        self.logger.info("Generating responses...")
        start_time = time.time()
        outputs = self.generate_responses(valid_prompts)
        responses = [output.outputs[0].text for output in outputs]
        self.logger.info(f"Generated {len(responses)} responses in {time.time() - start_time:.2f} seconds")

        # Create a list of responses that matches the original dataset length
        full_responses = []
        response_index = 0
        for prompt in prompts:
            if prompt is None:
                full_responses.append(None)
            else:
                full_responses.append(responses[response_index])
                response_index += 1

        self.logger.info(f"Number of rows in dataset: {len(dataset)}")
        self.logger.info(f"Number of generated responses: {len(full_responses)}")

        if len(dataset) == len(full_responses):
            dataset = dataset.add_column("test_response", full_responses)
            dataset = dataset.add_column("similar_prompts", similar_prompts)
            dataset = dataset.add_column("final_prompt", final_prompts)
            self.logger.info(f"Added {len(full_responses)} responses to the dataset")
        else:
            self.logger.error(f"Mismatch in number of rows: dataset has {len(dataset)}, but we have {len(full_responses)} responses")

        return dataset

if __name__ == "__main__":
    test_gen = MetaLearnKShotICLAlgorithm()
    updated_dataset = test_gen.generate_evaluation_responses(debug=True)
    logging.info(updated_dataset[0]["test_response"])