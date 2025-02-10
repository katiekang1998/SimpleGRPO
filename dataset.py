import torch
from typing import List, Tuple
import random
import numpy as np

# data generation code and prompt copied (with slight modification) from https://github.com/Jiayi-Pan/TinyZero
def gen_dataset(
    num_samples: int,
    num_operands: int = 6,
    max_target: int = 1000,
    min_number: int = 1,
    max_number: int = 100,
    operations: List[str] = ['+', '-', '*', '/'],
    seed_value: int = 42,
) -> List[Tuple]:
    """Generate dataset for countdown task.
    
    Args:
        num_samples: Number of samples to generate
        num_operands: Number of numbers provided in each sample
        max_target: Maximum value for target number
        min_number: Minimum value for provided numbers
        max_number: Maximum value for provided numbers
        operations: List of allowed operations
        seed_value: Random seed for reproducibility
        
    Returns:
        List of tuples containing (target, numbers, solution)
    """
    random.seed(seed_value)
    samples = []
    
    for _ in range(num_samples):
        # Generate random target
        target = random.randint(1, max_target)
        
        # Generate random numbers
        numbers = [random.randint(min_number, max_number) for _ in range(num_operands)]
        
        
        samples.append((target, numbers))
    
    return samples


def prepare_prompt(point):
    target = point[0]
    numbers = point[1]
    prompt = {}
    prompt["prompt"] = f"""<|start_header_id|>system<|end_header_id|>
    You are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer.<|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>
    Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    Let me solve this step by step.\n<think>"""
    prompt["numbers"] = numbers
    prompt["target"] = target
    return prompt


class CountdownDataset(torch.utils.data.Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]
    

def get_dataset(num_samples, max_target=100, max_number=10):
    dataset = gen_dataset(num_samples=num_samples, max_target=max_target, max_number=max_number)
    prompts = list(map(prepare_prompt, dataset))
    return CountdownDataset(prompts)
    
    
def custom_collate(batch):
    prompts = np.array([item["prompt"] for item in batch])
    numbers = np.stack([item["numbers"] for item in batch])
    targets = np.array([item["target"] for item in batch])
    
    return {
        "prompt": prompts,
        "numbers": numbers,
        "target": targets
    }

class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, batch):
        self.batch = batch
        # self.data = [torch.tensor(data_dict[k], dtype=torch.float32) for k in self.keys]
        # self.num_examples = self.data[0].shape[0]  # Assuming all arrays have the same num_examples size

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return {key: data[idx] for key, data in zip(self.keys, self.data)}