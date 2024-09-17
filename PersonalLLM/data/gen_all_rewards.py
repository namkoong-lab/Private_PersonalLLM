import os
import yaml
import torch
import argparse
import pandas as pd
from datasets import Dataset, load_dataset
from constants import REWARD_MODELS
from gen_score import score_dataframe
import logging

# Set up logging
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DotDict(dict):
    """Dictionary with dot notation access to attributes."""
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        del self[name]

def load_dataset_from_hub(dataset_name):
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    return dataset

def generate_all_rewards(args):
    dataset = load_dataset_from_hub(args.dataset_name)
    df = dataset.to_pandas()

    for reward_model in REWARD_MODELS.keys():
        logger.info(f"Processing reward model: {reward_model}")
        
        # Score the dataframe using the current reward model
        scored_df = score_dataframe(df, args.conf_filepath, reward_model)
        
        # Rename the score column to include the reward model name
        scored_df = scored_df.rename(columns={'score': f'score_{reward_model}'})
        
        # Merge the scores back into the original dataframe
        df = pd.merge(df, scored_df[[f'score_{reward_model}']], left_index=True, right_index=True)

    # Convert back to a Dataset object
    final_dataset = Dataset.from_pandas(df)

    # Push to Hub
    final_dataset.push_to_hub(args.output_dataset_name)
    logger.info(f"Dataset with all rewards pushed to: {args.output_dataset_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate rewards for all models and push dataset.")
    parser.add_argument("--conf_filepath", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process")
    parser.add_argument("--output_dataset_name", type=str, required=True, help="Name for the output dataset on HuggingFace Hub")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    with open(args.conf_filepath, "r") as file:
        config = yaml.safe_load(file)

    args = DotDict({**vars(args), **config})

    generate_all_rewards(args)
    logger.info("All rewards generated and dataset pushed successfully")
