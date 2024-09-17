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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LookUpMetaLearnKShotICLAlgorithm(BaseAlgorithm):
    """
    LookUpMetaLearnKShotICLAlgorithm is an algorithm that uses Look Up Methods to generate pairwise preference prompts.

    The end result is passing in self.num_shots (prompt, response) to the generation model.
    And we are given a meta-learning database.

    Step 1. We take the test row, which consists of the test user, past interactions (prompt, response_a, response_b, chosen_a/b), and the test prompt.
    Step 2. Now to find the similar users, We shortlist all the meta learning db to {n_similar_users_to_retrieve} rows of users, depending on whether there exists one prompt out of all prompts in the train row that is similar to the test prompt, using cosine similarity.
    Step 3. Now to rank these users from this shortlisted rows, we rank the users based on the cosine similarity between the test user past interactions (prompt, response) embedding and the meta learning user (prompt, response) embedding in the shortlisted rows. Pseudocode below:

        Pseudocode for if there are 3 test interactions, and 50 users in the meta-learning database
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        def compute_similarity_score(test_embeddings, train_embeddings):
            # Compute the cosine similarity matrix
            similarity_matrix = cosine_similarity(test_embeddings, train_embeddings)

            np.mean(np.max(similarity_matrix, axis=1))

        similarity_score = compute_similarity_score(test_embeddings, train_embeddings)
        print("Similarity Score:", similarity_score)

    Step 4. Based on this ranking, we select the top num_shots rows, which we then take their (prompt, response) that is similar to the test prompt and pass in to generation.
    """
    def __init__(self, args):
        super().__init__(args)
        self.logger = logging.getLogger(__name__)
        self.num_shots = args.num_shots
        self.args = args
        self.user_embedding_type = args.user_embedding_type
        self.user_embedding_computed_from_n_pr = args.user_embedding_computed_from_n_pr
        self.n_similar_users_to_retrieve = args.n_similar_users_to_retrieve
        
        self.logger.info("Loading datasets...")
        start_time = time.time()
        self.eval_dataset = load_dataset("andrewsiah/Eval_Pref_Dataset_with_stella_400M_v5_embeddings", cache_dir=args.cache_dir)['test']
        self.meta_learning_database = load_dataset(args.meta_learning_dataset, cache_dir=args.cache_dir)['train']
        self.logger.info(f"Datasets loaded in {time.time() - start_time:.2f} seconds")
        
        # Add debug logging
        self.logger.debug(f"Eval dataset columns: {self.eval_dataset.column_names}")
        self.logger.debug(f"First row of eval dataset: {self.eval_dataset[0]}")
        
        self.logger.info("Pre-computing user embeddings and data...")
        start_time = time.time()
        self.user_data_dict, self.all_prompt_embeddings, self.user_prompt_mapping = self.precompute_data()
        self.logger.info(f"Data pre-computed in {time.time() - start_time:.2f} seconds")

        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.max_history_tokens = 7000

    def precompute_data(self):
        user_data_dict = {}
        all_prompt_embeddings = []
        user_prompt_mapping = []

        for user in self.meta_learning_database:
            user_id = user['person_id']
            user_data_dict[user_id] = user
            
            for i in range(1, self.user_embedding_computed_from_n_pr + 1):
                prompt_embedding_key = f"prompt_{i}_embedding"
                if prompt_embedding_key in user:
                    all_prompt_embeddings.append(user[prompt_embedding_key])
                    user_prompt_mapping.append((user_id, i))

        return user_data_dict, np.array(all_prompt_embeddings), user_prompt_mapping

    @lru_cache(maxsize=1000)
    def find_users_with_similarity_to_test_prompt(self, test_prompt_embedding_tuple):
        """
        Find self.n_similar_users_to_retrieve number of users with most similar prompts to test prompt.

        Args:
            test_prompt_embedding_tuple (tuple): The embedding of the test prompt.

        Returns:
            tuple: A tuple of tuples, each containing (user_id, prompt_index, similarity_score)
                   for the top n_similar_users_to_retrieve users with similar prompts.
        """

        start_time = time.time()
        test_prompt_embedding = np.array(test_prompt_embedding_tuple)
        
        similarities = cosine_similarity([test_prompt_embedding], self.all_prompt_embeddings)[0]
        
        # Get top n_similar_users unique users
        sorted_indices = similarities.argsort()[::-1]
        unique_users = set()
        similar_users = []
        for idx in sorted_indices:
            user_id, prompt_index = self.user_prompt_mapping[idx]
            if user_id not in unique_users:
                unique_users.add(user_id)
                similar_users.append((user_id, prompt_index, similarities[idx]))
            if len(similar_users) == self.n_similar_users_to_retrieve:
                break

        self.logger.info(f"Found {len(similar_users)} users with similar prompts in {time.time() - start_time:.4f} seconds")
        return tuple(similar_users)

    def rank_similar_users(self, test_row, similar_users):
        """
        Rank similar users based on cosine similarity between test user past interactions (prompt, response) embedding and the meta learning user (prompt, response) embedding.

        Returns:
            list: A list of tuples, each containing (user_id, prompt_index, similarity_score)
        """
        start_time = time.time()
        test_embeddings = self.get_embeddings_from_row(test_row, is_test_row=True)
        
        user_scores = []
        for user_id, prompt_index, _ in similar_users:
            user_data = self.user_data_dict[user_id]
            user_embeddings = self.get_embeddings_from_row(user_data)
            
            if len(test_embeddings) > 0 and len(user_embeddings) > 0:
                similarity_matrix = cosine_similarity(test_embeddings, user_embeddings)
                similarity_score = np.mean(np.max(similarity_matrix, axis=1))
                user_scores.append((user_id, prompt_index, similarity_score))
            else:
                self.logger.warning(f"Skipping user {user_id} due to empty embeddings")
        
        ranked_users = sorted(user_scores, key=lambda x: x[2], reverse=True)
        self.logger.info(f"Ranked {len(ranked_users)} similar users in {time.time() - start_time:.4f} seconds")
        return ranked_users

    def get_embeddings_from_row(self, row, is_test_row=False):
        embeddings = []
        range_end = int(row['user_history_length']) if is_test_row else self.user_embedding_computed_from_n_pr + 1
        for i in range(1, range_end):
            chosen_key = f"chosen_{i}"
            pr_a_key = f"prompt_response_{i}_a_embedding"
            pr_b_key = f"prompt_response_{i}_b_embedding"
            
            if chosen_key not in row or pr_a_key not in row or pr_b_key not in row:
                continue
            
            chosen = row[chosen_key]
            winning_embedding = row[pr_a_key] if chosen == 'a' else row[pr_b_key]
            losing_embedding = row[pr_b_key] if chosen == 'a' else row[pr_a_key]
            
            if self.args.responses_to_include == 'winning_only':
                embeddings.append(winning_embedding)
            elif self.args.responses_to_include == 'losing_only' or self.args.responses_to_include == 'losing_as_winning_only':
                embeddings.append(losing_embedding)
            elif self.args.responses_to_include == 'winning_and_losing':
                embeddings.extend([winning_embedding, losing_embedding])
        return np.array(embeddings)

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
        try:
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
            
            # Get embeddings for the test prompt
            test_prompt_embedding = row['test_prompt_embedding']
            
            # Find users with similar prompts
            similar_users = self.find_users_with_similarity_to_test_prompt(tuple(test_prompt_embedding))
            self.logger.debug(f"Found similar users in {time.time() - start_time:.4f} seconds")
            
            start_time = time.time()
            ranked_users = self.rank_similar_users(row, similar_users)
            self.logger.debug(f"Ranked similar users in {time.time() - start_time:.4f} seconds")
            
            # Select top num_shots users and their prompts
            selected_users = ranked_users[:self.num_shots]

            # Log similar prompts
            similar_prompts = []
            for user_id, similar_prompt_index, _ in selected_users:
                similar_prompt = self.user_data_dict[user_id][f"prompt_{similar_prompt_index}"]
                similar_prompts.append(f"User {user_id}, Prompt: {similar_prompt}")

            total_tokens = 0
            for user_id, similar_prompt_index, _ in selected_users:
                similar_prompt = self.user_data_dict[user_id][f"prompt_{similar_prompt_index}"]
                winning_response, losing_response = self.find_winning_losing_responses(user_id, similar_prompt_index)
                
                if self.args.responses_to_include == 'winning_only':
                    example = f"User: {similar_prompt}\nLiked Response: {winning_response}\n\n"
                elif self.args.responses_to_include == 'losing_only':
                    example = f"User: {similar_prompt}\nDisliked Response: {losing_response}\n\n"
                elif self.args.responses_to_include == 'losing_as_winning_only':
                    example = f"User: {similar_prompt}\nLiked Response: {losing_response}\n\n"
                elif self.args.responses_to_include == 'winning_and_losing':
                    example = f"User: {similar_prompt}\nLiked Response: {winning_response}\nDisliked Response: {losing_response}\n\n"
                
                example_tokens = len(self.tokenizer.encode(example))
                if total_tokens + example_tokens > self.max_history_tokens:
                    self.logger.warning(f"WARNING: Dropped {len(selected_users) - len(history)}/{self.num_shots} shot ICL examples")
                    break
                
                history.append(example)
                total_tokens += example_tokens
            
            prompt += "".join(history)
            if self.args.responses_to_include == 'winning_only':
                prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should be similar to the winning responses. "
            elif self.args.responses_to_include == 'losing_only':
                prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should be dissimilar to the losing responses. "
            elif self.args.responses_to_include == 'losing_as_winning_only':
                prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should be similar to the winning responses. "
            elif self.args.responses_to_include == 'winning_and_losing':
                prompt += "Use the contexts above to generate a good response for the user prompt below. Your response should be similar to the winning responses and different from the losing responses. "
            # TODO: Test Prompt is wrong!
            # Add debug logging
            self.logger.debug(f"!!!Row keys: {row.keys()}")
            self.logger.debug(f"!!!Test prompt: {row.get('test_prompt', 'Not found')}")
            self.logger.debug(f"!!!Best response: {row.get('best_response', 'Not found')}")
            
            prompt += f"User: {row['test_prompt']} \nResponse: "

            # Log the final prompt
            self.logger.debug("Final prompt passed to the language model:")
            self.logger.debug(prompt)
            
            self.logger.debug(f"Generated pairwise preference prompt in {time.time() - start_time:.4f} seconds")
            return prompt, "\n".join(similar_prompts), prompt
        except Exception as e:
            self.logger.error(f"Error in generate_pairwise_pref_prompt: {str(e)}")
            self.logger.error(f"Row data: {row}")
            raise

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
        self.logger.info(f"Generated {len(prompts)} prompts in {time.time() - start_time:.2f} seconds")

        self.logger.info("Generating responses...")
        start_time = time.time()
        valid_prompts = [p for p in prompts if p is not None]
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
            # You might want to handle this error case, e.g., by truncating or padding the responses

        # if args.csv_output:
        #     df = dataset.to_pandas()
        #     df.to_csv(args.csv_output, index=False)
        #     self.logger.info(f"Wrote dataset with debug information to {args.csv_output}")
        return dataset

if __name__ == "__main__":
    test_gen = LookUpMetaLearnKShotICLAlgorithm()
    updated_dataset = test_gen.generate_evaluation_responses(debug=True)
    logging.info(updated_dataset[0]["test_response"])