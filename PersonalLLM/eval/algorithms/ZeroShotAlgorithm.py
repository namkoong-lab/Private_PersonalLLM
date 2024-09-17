import random
import string
from algorithms.BaseAlgorithm import BaseAlgorithm
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams


class ZeroShotAlgorithm(BaseAlgorithm):
    """
    This does not utilize the meta_learning_database, and generates from vllm using previous rows only.

    """

    def __init__(self, args):
        super().__init__(args)

    def generate_prompt(self, row: dict) -> str:
        prompt = "User: " + row['test_prompt']
        return prompt

    def generate_responses(self, prompts):
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=512)
        outputs = self.llm.generate(prompts, sampling_params)
        return outputs

    def generate_evaluation_responses(self, args) -> Dataset:
        debug = args.debug
        # dataset = self.eval_dataset
        dataset = load_dataset("andrewsiah/Eval_Pref_Dataset_with_stella_400M_v5_embeddings", cache_dir=args.cache_dir)['test']
        if debug:
            dataset = dataset.select(range(50))
        prompts = [self.generate_prompt(dataset[i]) for i in range(len(dataset))]
        outputs = self.generate_responses(prompts)
        responses = [output.outputs[0].text for output in outputs]
        dataset = dataset.add_column("test_response", responses)
        return dataset

if __name__ == "__main__":
    test_gen = ZeroShotAlgorithm()
    updated_dataset = test_gen.generate_evaluation_responses(debug=True)
    print(updated_dataset[0]["test_response"])
