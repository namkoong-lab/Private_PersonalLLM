import subprocess
import itertools
import logging
import os
from datetime import datetime

"""
Todo:

"""

# Define debug mode
debug = False
# Define the parameters to sweep through
params = {
    "seeds": [27],
    "num_shots": [1,5],
    "responses_to_include": ["losing_only", "losing_as_winning_only"],
    "user_embedding_types": [],
    "user_embedding_computed_from_n_pr": [],
    "n_similar_users_to_retrieve": [],
    "meta_learning_datasets": [
        # "andrewsiah/MetaLearningPairwisePrefDB_Argmax_1000",
        # "andrewsiah/MetaLearningPairwisePrefDB_Argmax_10000",
        # "andrewsiah/MetaLearningPairwisePrefDB_Argmax_50000"
    ],
    "algorithms_eval_types": {
        "KShotICLAlgorithm": ["pairwise_pref"],
        # "MetaLearnKShotICLAlgorithm": ["pairwise_pref"],
        # "SelfKShotICLAlgorithm": ["pairwise_pref"],
        # "LookUpMetaLearnKShotICLAlgorithm": ["pairwise_pref"]
    },
    "log_levels": ["INFO"]
}


# Generate all combinations of non-empty parameters
non_empty_params = [
    params["seeds"],
    params["num_shots"],
    params["responses_to_include"],
    params["user_embedding_types"] or [None],
    params["user_embedding_computed_from_n_pr"] or [None],
    params["n_similar_users_to_retrieve"] or [None],
    params["meta_learning_datasets"] or [None], 
    params["algorithms_eval_types"].items(),
    params["log_levels"]
]

param_combinations = list(itertools.product(*non_empty_params))

# Loop through all parameter combinations
for seed, shots, response_type, user_embedding_type, n_pr, n_similar_users, meta_learning_dataset, (algorithm, eval_types), log_level in param_combinations:
    for eval_type in eval_types:
        command = [
            "python",
            "eval.py",
            "--algorithm=" + algorithm,
            "--seed=" + str(seed),
            f"--log_level={log_level}"
        ]
        if shots is not None:
            command.append("--num_shots=" + str(shots))
        if response_type is not None:
            command.append("--responses_to_include=" + response_type)
        if meta_learning_dataset is not None:
            command.append("--meta_learning_dataset=" + meta_learning_dataset)
        if user_embedding_type:
            command.append("--user_embedding_type=" + user_embedding_type)
        if n_pr is not None:
            command.append("--user_embedding_computed_from_n_pr=" + str(n_pr))
        if n_similar_users is not None:
            command.append("--n_similar_users_to_retrieve=" + str(n_similar_users))
        if eval_type:
            command.append("--eval_type=" + eval_type)
        if debug:
            command.append("--debug")
            # Create a unique filename for the debug output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_filename = f"debug_output_{algorithm}_{eval_type}_{timestamp}.txt"
            
            try:
                with open(debug_filename, 'w') as debug_file:
                    subprocess.check_call(command, stdout=debug_file, stderr=subprocess.STDOUT)
                print(f"Debug output written to {debug_filename}")
            except Exception as e:
                logging.error(f"Failed to run {algorithm} with {eval_type if eval_type else ''}, "
                              f"seed {seed}, {shots} shots, {response_type} responses, "
                              f"user_embedding_type {user_embedding_type}, "
                              f"user_embedding_computed_from_n_pr {n_pr}, "
                              f"n_similar_users_to_retrieve {n_similar_users}, "
                              f"meta_learning_dataset {meta_learning_dataset}, "
                              f"log_level {log_level}: {str(e)}")
        else:
            try:
                subprocess.check_call(command)
            except Exception as e:
                logging.error(f"Failed to run {algorithm} with {eval_type if eval_type else ''}, "
                              f"seed {seed}, {shots} shots, {response_type} responses, "
                              f"user_embedding_type {user_embedding_type}, "
                              f"user_embedding_computed_from_n_pr {n_pr}, "
                              f"n_similar_users_to_retrieve {n_similar_users}, "
                              f"meta_learning_dataset {meta_learning_dataset}, "
                              f"log_level {log_level}: {str(e)}")