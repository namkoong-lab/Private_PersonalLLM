import logging
import os

log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

import time
import torch
import random
import argparse
import importlib
import numpy as np
import pandas as pd
from typing import Type
import multiprocessing as mp
from constants import REWARD_MODELS
from algorithms.BaseAlgorithm import BaseAlgorithm

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'algorithms')
for file in os.listdir(directory):
    if file.endswith('.py') and file != '__init__.py' and file != 'BaseAlgorithm.py':
        module_name = file[:-3]  # removes the .py at the end
        module = importlib.import_module('algorithms.' + module_name)
        globals()[module_name] = module

reward_models_names = sorted(list(REWARD_MODELS.keys()))

def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def persona_score_dataset(dataset_with_response, args):
    from gen_score import score_dataframe

    results_dict = {}
    results_df = pd.DataFrame(dataset_with_response)
    results_df.rename(columns={'test_prompt': 'prompt', 'test_response': 'response'}, inplace=True)

    for reward_model in reward_models_names:
        scored_df = score_dataframe(results_df, args.conf_filepath, reward_model)

        results_df['score_' + reward_model] = scored_df['score']

        mean_score = REWARD_MODELS[reward_model]['mean']
        std_score = REWARD_MODELS[reward_model]['sd']
        results_df['score_' + reward_model] = (results_df['score_' + reward_model] - mean_score) / std_score
        model_mean = results_df['score_' + reward_model].mean()
        results_dict[f'mean_test_{reward_model}'] = model_mean
        results_dict[f'mean_train_{reward_model}'] = mean_score
        results_dict[f'std_train_{reward_model}'] = std_score
        logging.info(f"Mean for {reward_model}: {model_mean}; Train Set Mean: {mean_score}, Train Set Std: {std_score}")

    for i, row in results_df.iterrows():
        reward_scores = [row[f'score_{model_name}'] for model_name in reward_models_names]
        results_df.loc[i, 'persona_weighted_score'] = np.dot(row['person_weight'][:len(reward_models_names)], reward_scores)

    results_dict['total_score'] = results_df['persona_weighted_score'].mean()
    results_dict['win_rate_against_bestresponse'] = len(results_df[results_df['persona_weighted_score'] > results_df['best_response_reward']]) / len(results_df) * 100
    results_dict['win_rate_against_gpt4o'] = len(results_df[results_df['persona_weighted_score'] > results_df['gpt4o_reward']]) / len(results_df) * 100

    logging.info(f"Total Score: {results_dict['total_score']}")
    logging.info(f"Win Rate Against Best_Response_Reward: {results_dict['win_rate_against_bestresponse']}%")
    logging.info(f"Win Rate Against gpt4o: {results_dict['win_rate_against_gpt4o']}%")

    return results_dict, results_df

def evaluate_response_algorithm(response_algorithm_class: Type[BaseAlgorithm]) -> float:
    response_algorithm = response_algorithm_class(args)
    responsed_dataset = response_algorithm.generate_evaluation_responses(args)
    responded_df = pd.DataFrame(responsed_dataset)
    
    results, scored_df = persona_score_dataset(responded_df, args)
    # if args.debug:
    scored_df.to_csv(args.csv_output, index=False)
    return results


if __name__ == "__main__":
    start_time = time.time()

    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_filepath", type=str, default="../eval/eval_conf.yaml", help="Path to the configuration file")
    parser.add_argument("--cache_dir", type=str, default="/shared/share_mala/andrew/huggingface/cache", help="Path to the cache directory")
    parser.add_argument("--csv_output", type=str, default=None, help="Path to the output CSV file")
    parser.add_argument("--debug", action='store_true', default=False, help="Enable debug mode")
    parser.add_argument("--gpu", type=int, default=0, help="GPU CUDA number to use for computation")
    parser.add_argument("--algorithm", type=str, default="TestAlgorithm", help="Name of the response algorithm to use")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation")
    parser.add_argument("--eval_type", type=str, default="pairwise_pref", choices=["reward", "pairwise_pref"], help="Type of evaluation to perform: 'reward' or 'pref'")
    parser.add_argument("--vllm_gpu_utilization_pct", type=int, default=0.5, help="Percentage of GPU utilization for VLLM. Default is 0.5")
    parser.add_argument("--vllm_model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name for VLLM")
    # Below are arguments for ICL and MetaLearning
    parser.add_argument("--num_shots", type=int, default=0, help="Number of shots for ICL")
    parser.add_argument("--responses_to_include", choices=['none', 'winning_only', 'losing_only','losing_as_winning_only', 'winning_and_losing'], default='none', help="Responses from dataset to include in the prompt")
    parser.add_argument("--user_embedding_type", choices=['winning', 'losing', 'win_minus_lose'], default='winning', help="Type of user embedding to use")
    parser.add_argument("--user_embedding_computed_from_n_pr", type=int, default=10, choices=[3, 5, 10, 50], help="Number of prompts to compute user embedding from")
    parser.add_argument("--n_similar_users_to_retrieve", type=int, default=10, help="Number of similar users to retrieve")
    parser.add_argument("--meta_learning_dataset", type=str, default='namkoong-lab/PersonalLLM_InteractionDatabase', help="Path to the meta learning dataset")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")
    args = parser.parse_args()

    # Update the log level based on the parsed arguments
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    set_seed(args.seed)

    from datetime import datetime
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
    default_output_path = f'scores/score_df_{args.algorithm}_{current_datetime}.csv'
    if args.csv_output is None:
        args.csv_output = default_output_path 
    # imports the class from the module
    algorithm_class = getattr(globals()[args.algorithm], args.algorithm)

    if args.debug:
        reward_models_names = ['oasst_pythia_1b']
        pass

    results = evaluate_response_algorithm(algorithm_class)
    end_time = time.time()
    time_taken = end_time - start_time
    logging.info(f"Start time: {start_time}, Time taken: {time_taken}")

    if os.path.isfile('scores/all_scores.csv') and os.path.getsize('scores/all_scores.csv') > 0:
        df = pd.read_csv('scores/all_scores.csv')
    else:
        df = pd.DataFrame(
            columns=[
                "datetime",
                "eval_type",
                "algorithm",
                "num_shots",
                "responses_to_include",
                "debug",
                "total_score",
                "win_rate_against_bestresponse",
                "win_rate_against_gpt4o",
                "time_taken",
                "seed",
                "conf_filepath",
                "csv_output",
                "gpu",
                "vllm_gpu_utilization_pct",
                "vllm_model_name",
                "user_embedding_type",
                "user_embedding_computed_from_n_pr",
                "n_similar_users_to_retrieve",
                "meta_learning_dataset"
            ]
        )

    new_row = {
        'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'eval_type': args.eval_type,
        'algorithm': args.algorithm,
        'num_shots': args.num_shots,
        'responses_to_include': args.responses_to_include,
        'debug': args.debug,
        'total_score': results['total_score'],
        'win_rate_against_bestresponse': results['win_rate_against_bestresponse'],
        'win_rate_against_gpt4o': results['win_rate_against_gpt4o'],
        'time_taken': time_taken,
        'seed': args.seed,
        'conf_filepath': args.conf_filepath,
        'csv_output': args.csv_output,
        'gpu': args.gpu,
        'vllm_gpu_utilization_pct': args.vllm_gpu_utilization_pct,
        'vllm_model_name': args.vllm_model_name,
        'user_embedding_type': args.user_embedding_type,
        'user_embedding_computed_from_n_pr': args.user_embedding_computed_from_n_pr,
        'n_similar_users_to_retrieve': args.n_similar_users_to_retrieve,
        'meta_learning_dataset': args.meta_learning_dataset
    }
    
    for model in reward_models_names:
        new_row[f'mean_test_{model}'] = results[f'mean_test_{model}']
        new_row[f'mean_train_{model}'] = results[f'mean_train_{model}']
        new_row[f'std_train_{model}'] = results[f'std_train_{model}']

    df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    df.to_csv('scores/all_scores.csv', index=False)