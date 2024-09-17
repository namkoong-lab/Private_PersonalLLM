import os
import re
import json
import yaml
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from multiprocessing import Process, Manager
import multiprocessing
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, pipeline, PreTrainedTokenizer
from fastchat.conversation import get_conv_template, Conversation
from rewardbench import (
    DPO_MODEL_CONFIG,
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
)
from constants import REWARD_MODELS
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
    def __getstate__(self):
        return self
    def __setstate__(self, state):
        self.update(state)

def load_and_combine_datasets(dataset_names):
    """Load datasets and combine them into a single dataset."""
    logger.info(f"Loading and combining datasets: {dataset_names}")
    datasets = [load_dataset(name)['train'] if 'train' in load_dataset(name) else load_dataset(name) for name in dataset_names]
    combined_data = {key: sum([list(dataset[key]) for dataset in datasets], []) for key in datasets[0].features}
    logger.info(f"Combined dataset created with {len(combined_data[list(combined_data.keys())[0]])} samples")
    return Dataset.from_dict(combined_data)

def init_models(args, model_builder, pipeline_builder, quantized, tokenizer, device):
    logger.info(f"Initializing models on device: {device}")
    if quantized:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": {"": device},
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            "device_map": {"": device},
            }

    model = model_builder(args.model_name, **model_kwargs, cache_dir=args.cache_dir,trust_remote_code=args.trust_remote_code)
    model.eval()
    reward_pipe = pipeline_builder(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )

    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id

    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True

    logger.info("Models initialized successfully")
    return tokenizer, reward_pipe

def generate_reward_model_formatted_responses(dataset, tokenizer: PreTrainedTokenizer = None, conv: Conversation = None):
    logger.info("Generating reward model formatted responses")
    stylized_columns = {}
    updated_rows = []
    if tokenizer is None and tokenizer.chat_template is None and not conv:
        logger.warning("Warning: No tokenizer chat template or fastchat conversation passed. Using Default Template. Please check if that's alright for current model.")
    for row in tqdm(dataset, desc="Processing rows"):
        new_row = row.copy()
        for key, value in row.items():
            if key == "response":
                if tokenizer and tokenizer.chat_template:
                    stylized_response = tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": value},
                        ],
                        tokenize=False,
                        add_generation_prompt=False,
                    ).replace(tokenizer.bos_token, "")
                elif conv:
                    # AS: 3 models in reward bench takes in a conv when chat_template is None.
                    # Refer to utils.py in rewardbench (prepare_dialogue function) for `conv`
                    conv.messages = [
                        [conv.roles[0], row["prompt"]],
                        [conv.roles[1], value],
                    ]
                    stylized_response = conv.get_prompt()
                else:
                    stylized_response = f"User: {row['prompt']}\nAssistant: {value}"
                if key not in stylized_columns:
                    stylized_columns[key] = []
                stylized_columns[key].append(stylized_response)
                new_row["rformatted_promptresponse"] = stylized_response
        updated_rows.append(new_row)
    dataset = Dataset.from_dict({key: [dic[key] for dic in updated_rows] for key in updated_rows[0]})
    logger.info(f"Generated {len(dataset)} formatted responses")
    return dataset

def compute_rewards(args, new_dataset, start, end, batch_size, model_pipeline, pipe_kwargs):
    logger.info(f"Computing rewards for samples {start} to {end}")
    rewarded_dataset = {}

    column_name = 'rformatted_promptresponse'
    rewards_column_name = 'reward'
    rewards = []
    for batch_start in trange(start, end, batch_size, desc="Batch:"):
        batch_end = min(batch_start + batch_size, end)
        batch_texts = new_dataset[batch_start:batch_end][column_name]

        pipe_outputs = model_pipeline(batch_texts, **pipe_kwargs)
        # Everything is good after this.
        if args.debug:
            with open("pipe_outputs.txt", "a") as f:
                for text in batch_texts:
                    f.write(str(text) + '\n')
                f.write(str(pipe_outputs) + "\n" + "-"*40 + "\n")
        if isinstance(pipe_outputs[0], dict):
            batch_rewards = [output["score"] for output in pipe_outputs]
        elif isinstance(pipe_outputs[0][0], dict):
            batch_rewards = [output[0]["score"] for output in pipe_outputs]
        else:
            batch_rewards = [output[0].cpu().numpy().tolist() for output in pipe_outputs]
        
        rewards.extend(batch_rewards)
        if args.debug:
            with open("pipe_outputs.txt", "a") as f:
                f.write(f"BATCH_REWARDS: {str(batch_rewards)}\n")
                f.write(f"REWARDS: {str(rewards)}\n")
    
    rewarded_dataset[rewards_column_name] = rewards
    for column in new_dataset.features:
        rewarded_dataset[column] = new_dataset[column][start:end] 
    logger.info(f"Computed rewards for {len(rewards)} samples")
    return Dataset.from_dict(rewarded_dataset)

def worker(gpu, start_idx, end_idx, return_dict, worker_id, args, model_builder, pipeline_builder, quantized, dataset):
    logger.info(f"Worker {worker_id} started on GPU {gpu}")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Worker {worker_id} started (attempt {attempt + 1})")
            device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
            tokenizer_path = args.tokenizer if args.tokenizer else args.tokenizer_name
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
            tokenizer, model_pipeline = init_models(args, model_builder, pipeline_builder, quantized, tokenizer, device)
            new_dataset = generate_reward_model_formatted_responses(dataset, tokenizer)

            pipe_kwargs = {
                "batch_size": args.batch_size,
                "truncation": True,
                "padding": "longest",
                "max_length": args.max_length,
                "function_to_apply": "none",
                "return_token_type_ids": False,
            }
            rewarded_dataset = compute_rewards(args, new_dataset, start_idx, end_idx, args.batch_size, model_pipeline, pipe_kwargs)
            logger.info(f"Worker {worker_id} completed successfully")
            return_dict[worker_id] = rewarded_dataset
            return  # Exit the function if successful
        except Exception as e:
            logger.error(f"Worker {worker_id} failed on attempt {attempt + 1} with error: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"Worker {worker_id} failed after {max_retries} attempts")

def score_dataframe(to_do_df: pd.DataFrame, config_fp:str, reward_model:str, max_retries: int = 3) -> pd.DataFrame:
    """
    Takes in a DataFrame, adds a score column using the reward model, and returns the updated DataFrame.
    
    Args:
        to_do_df (pd.DataFrame): DataFrame containing the data to be scored. 
                                 Expected columns: ["data_idx", "n_rag", "choice", "prompt", "response"]
        config_fp (str): Filepath to the configuration YAML file containing parameters for scoring.
        reward_model (str, optional): The reward model to be used for scoring. If None, it will be read from the config file.
        max_retries (int, optional): The maximum number of retries for failed workers. Defaults to 3.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'score' column.
    """

    logger.info(f"Scoring dataframe using reward model: {reward_model}")
    
    with open(config_fp, 'r') as file:
        config = yaml.safe_load(file)

    args = DotDict(config)

    if reward_model in REWARD_MODELS:
        args.model_name = REWARD_MODELS[reward_model]["model_name"]
        args.tokenizer_name = REWARD_MODELS[reward_model]["tokenizer_name"]
        args.chat_template = REWARD_MODELS[reward_model]["fastchat_chat_template"]
    else:
        logger.error(f"Reward model '{reward_model}' not found in REWARD_MODELS.")
        raise ValueError(f"Reward model '{reward_model}' not found in REWARD_MODELS.")

    dataset = Dataset.from_pandas(to_do_df)

    device_count = len(args.gpus)
    total_samples = len(dataset)
    samples_per_gpu = total_samples // device_count

    logger.info(f"Using {device_count} GPUs for {total_samples} samples")

    is_dpo = False
    MODEL_CONFIGS = REWARD_MODEL_CONFIG

    if args.chat_template:
        from fastchat.conversation import get_conv_template
        conv = get_conv_template(args.chat_template)
    else:
        conv = None

    if args.model_name in MODEL_CONFIGS:
        config = MODEL_CONFIGS[args.model_name]
    else:
        config = MODEL_CONFIGS["default"]

    quantized = config["quantized"]  # only Starling isn't quantized for now
    custom_dialogue = config["custom_dialogue"]
    pipeline_builder = config["pipeline_builder"]
    model_builder = config["model_builder"]

    logger.info(f"PIPELINE_BUILDER: {pipeline_builder}")
    logger.info(f"MODEL_BUILDER: {model_builder}")

    if custom_dialogue:
        logger.error("Custom dialogue not implemented yet for simpler data formatting.")
        raise NotImplementedError("Custom dialogue not implemented yet for simpler data formatting.")

    processes = []
    manager = Manager()
    return_dict = manager.dict()

    for i, gpu in enumerate(args.gpus):
        gpu_row_start = args.start + i * samples_per_gpu
        gpu_row_end = args.start + (i + 1) * samples_per_gpu if i < device_count - 1 else len(dataset)
        p = Process(target=worker, args=(gpu, gpu_row_start, gpu_row_end, return_dict, i, args, model_builder, pipeline_builder, quantized, dataset))
        processes.append(p)

    try:
        for p in processes:
            logger.info(f"Starting process: {p.name}")
            p.start()

        for p in processes:
            logger.info(f"Joining process: {p.name}")
            p.join()
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
    finally:
        for p in processes:
            if p.is_alive():
                logger.warning(f"Terminating process {p.pid}")
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    logger.warning(f"Process {p.pid} did not terminate, killing it")
                    p.kill()

    all_rewarded_datasets = []
    for i in range(len(args.gpus)):
        try:
            all_rewarded_datasets.append(return_dict[i])
        except KeyError:
            logger.error(f"Error: Worker {i} did not return a dataset.")

    if all_rewarded_datasets:
        concatenated_dataset = Dataset.from_dict(
            {
                column: sum([list(ds[column]) for ds in all_rewarded_datasets], [])
                for column in all_rewarded_datasets[0].features
            }
        )
    else:
        concatenated_dataset = Dataset.from_dict({})
        logger.error("No datasets were successfully processed")
        raise RuntimeError("No datasets were successfully processed")

    concatenated_df = concatenated_dataset.to_pandas()
    concatenated_df.rename(columns={"reward": "score"}, inplace=True)
    concatenated_df.drop(columns=["rformatted_promptresponse"], inplace=True)
    torch.cuda.empty_cache()
    logger.info(f"Scoring completed. Dataframe shape: {concatenated_df.shape}")
    return concatenated_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")
    args = parser.parse_args()

    # Update the log level based on the parsed arguments
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    logger.info("Starting gen_score.py")