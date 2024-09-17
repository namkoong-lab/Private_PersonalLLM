from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams

class BaseAlgorithm:
    def __init__(self, args):
        # TODO: Update the datasets. 
        if args.eval_type == "pairwise_pref":
            self.eval_dataset = load_dataset("andrewsiah/Eval_Pref_Dataset_with_stella_400M_v5_embeddings", cache_dir=args.cache_dir)['test']
        if args.eval_type == "reward":
            self.eval_dataset = load_dataset("andrewsiah/Eval_Reward_Dataset", cache_dir=args.cache_dir)["test"]
            self.meta_learning_database = load_dataset(
                "andrewsiah/MetaLearningPrefDatabase", cache_dir=args.cache_dir
            )["train"]

        self.llm = LLM(
            model=args.vllm_model_name,
            trust_remote_code=True,
            tensor_parallel_size=1,
            download_dir="/shared/share_mala/andrew/huggingface/cache",
            disable_log_stats=True,
            gpu_memory_utilization=args.vllm_gpu_utilization_pct,
        )

    def generate_evaluation_responses(self, args) -> Dataset:
        raise NotImplementedError
