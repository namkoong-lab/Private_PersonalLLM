import numpy as np
from numba import jit, prange
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, load_dataset
import argparse
from constants import REWARD_MODELS
from scipy.stats import dirichlet

parser = argparse.ArgumentParser()
parser.add_argument("--n_persons", type=int, default=50000, help="Number of personas to generate")
parser.add_argument("--interaction_length", type=int, default=10, help="Number of interactions per person")
parser.add_argument("--alpha", type=float, default=0.05, help="Alpha value for generating personas")
parser.add_argument("--push_to_hub_path", type=str, default="MetaLearningPairwisePrefDB", help="Path to push the huggingface dataset")
parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation")
parser.add_argument("--argmax", default=True, action="store_true", help="Use argmax/argmin selection instead of random")

args = parser.parse_args()
dataset = load_dataset("andrewsiah/PersonalLLM_with_stella_400M_v5_embeddings", split="train")
ds = {key: np.array(dataset[key]) for key in dataset.features.keys()}

reward_models_names = sorted(list(REWARD_MODELS.keys()))
n_rm = len(reward_models_names)


# Define functions for calculating distances and filtering similar personas
@jit(nopython=True, parallel=True)
def calculate_distances(persons):
    n = len(persons)
    distances = np.empty((n, n), dtype=np.float64)
    for i in prange(n):
        for j in prange(i + 1, n):
            distances[i, j] = np.sqrt(np.sum((persons[i] - persons[j]) ** 2))
            distances[j, i] = distances[i, j]
    return distances

@jit(nopython=True)
def filter_most_different_personas(persons, n_keep):
    # Keep the n_keep most different personas
    distances = calculate_distances(persons)
    total_distances = np.sum(distances, axis=1)
    sorted_indices = np.argsort(total_distances)[::-1]
    keep_indices = sorted_indices[:n_keep]
    
    return persons[keep_indices]

# Define function for generating personas
def generate_personas(
    alpha_values,
    n_rm,
    n_persons,
    n_keep,
    same_alpha=True,
    random_alpha=False,
):
    all_persons = []
    random_state = args.seed
    if same_alpha:
        for alpha in alpha_values:
            alphas = np.array([alpha] * n_rm)
            persons = dirichlet.rvs(alphas, size=n_persons, random_state=random_state)
            all_persons.append(persons)
    if random_alpha:
        alphas = np.random.choice(alpha_values, size=n_rm)
        persons = dirichlet.rvs(alphas, size=n_persons, random_state=random_state)
        all_persons.append(persons)
    all_persons = np.vstack(all_persons)
    return all_persons

# Generate personas
n_persons = args.n_persons
n_responses = 8 # The number of responses for each prompt in the original dataset
alpha_values = [args.alpha]
persons = generate_personas(
    alpha_values,
    n_rm,
    n_persons,
    n_persons,
    same_alpha=True,
    random_alpha=False,
)

# Pre-compute random indices and choices
# Set seed for reproducibility
np.random.seed(args.seed)
database_index = np.random.randint(0, high=len(ds['prompt']), size=(len(persons), args.interaction_length))
if not args.argmax:
    response_choices = np.array([np.random.choice(np.arange(1, n_responses + 1), size=2, replace=False) 
                                 for _ in range(len(persons) * args.interaction_length)]).reshape(len(persons), args.interaction_length, 2)

@jit(nopython=True)
def calculate_rewards(person_weights, response_data):
    # Ensure person_weights is a 1D array
    person_weights = person_weights.ravel()
    # Ensure response_data is a 2D array with the same number of columns as person_weights
    response_data = response_data.reshape(-1, len(person_weights))
    return np.sum(person_weights * response_data, axis=1)

historical_data = []

for i in tqdm(range(len(persons))):
    person_data = {
        "person_id": i,
        "person_weight": persons[i].tolist(),
    }

    winning_embeddings = []
    losing_embeddings = []
    direction_embeddings = []
    
    for j in range(args.interaction_length):
        prompt_n = database_index[i, j]
        
        if args.argmax:
            # Calculate rewards for all responses
            response_data = np.array([
                [ds[f'response_{k}_{model_name}'][prompt_n] for model_name in reward_models_names]
                for k in range(1, n_responses + 1)
            ])
            
            rewards = calculate_rewards(persons[i], response_data)
            
            # Get argmax and argmin
            argmax = np.argmax(rewards)
            argmin = np.argmin(rewards)
            
            # Randomly shuffle argmax and argmin
            if np.random.random() < 0.5:
                response_1, response_2 = argmax + 1, argmin + 1
                chosen = "a"
            else:
                response_1, response_2 = argmin + 1, argmax + 1
                chosen = "b"
        else:
            response_1, response_2 = response_choices[i, j]
            
            response_data = np.array([
                [ds[f'response_{response_1}_{model_name}'][prompt_n] for model_name in reward_models_names],
                [ds[f'response_{response_2}_{model_name}'][prompt_n] for model_name in reward_models_names]
            ])
            
            rewards = calculate_rewards(persons[i], response_data)
            
            reward_1, reward_2 = rewards[0], rewards[1]
            
            chosen = "a" if np.sum(reward_1) > np.sum(reward_2) else "b"
        
        person_data[f"prompt_{j+1}"] = ds["prompt"][prompt_n]
        person_data[f"response_{j+1}_a"] = ds[f"response_{response_1}"][prompt_n]
        person_data[f"response_{j+1}_b"] = ds[f"response_{response_2}"][prompt_n]
        person_data[f'chosen_{j+1}'] = chosen
        person_data[f"prompt_response_{j+1}_a_embedding"] = ds[f"prompt_response_{response_1}_embedding"][prompt_n]
        person_data[f"prompt_response_{j+1}_b_embedding"] = ds[f"prompt_response_{response_2}_embedding"][prompt_n]
        person_data[f"prompt_{j+1}_embedding"] = ds["prompt_embedding"][prompt_n]

        chosen_embedding = ds[f"prompt_response_{response_1}_embedding"][prompt_n] if chosen == "a" else ds[f"prompt_response_{response_2}_embedding"][prompt_n]
        losing_embedding = ds[f"prompt_response_{response_2}_embedding"][prompt_n] if chosen == "a" else ds[f"prompt_response_{response_1}_embedding"][prompt_n]
        
        winning_embeddings.append(chosen_embedding)
        losing_embeddings.append(losing_embedding)
        direction_embeddings.append(chosen_embedding - losing_embedding)
    
    for n in [3, 5, 10]:
        if len(direction_embeddings) >= n:
            person_data[f"user_embedding_avg_win_minus_lose_{n}"] = np.mean(direction_embeddings[:n], axis=0).tolist()
            person_data[f"user_embedding_avg_winning_{n}"] = np.mean(winning_embeddings[:n], axis=0).tolist()
            person_data[f"user_embedding_avg_losing_{n}"] = np.mean(losing_embeddings[:n], axis=0).tolist()
    historical_data.append(person_data)

historical_database = pd.DataFrame(historical_data)
historical_database_hf = Dataset.from_pandas(historical_database)
# Determine the push to hub path based on number of personas and argmax flag
push_to_hub_path = args.push_to_hub_path

if args.argmax:
    push_to_hub_path += "_Argmax"

push_to_hub_path += f"_{args.n_persons}"

print(f"Pushing dataset to hub: {push_to_hub_path}")

historical_database_hf.push_to_hub(push_to_hub_path)