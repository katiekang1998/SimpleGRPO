from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig
import wandb
from tqdm import tqdm
import time
from datetime import datetime
from accelerate import Accelerator

from dataset import get_dataset, custom_collate
from reward import get_sample_stats, get_rewards
from policy_reinforce import PolicyREINFORCE


def main():
    ### Parameters ###
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    project_name = "CountdownRL"
    run_name = "REINFORCE"

    learning_rate = 1e-5
    grad_clip_norm = 1
    
    num_train_steps = 10000
    num_steps_per_train_log = 1
    num_steps_per_test_log = 100

    num_samples_train = 16000
    num_samples_test = 128

    batch_size = 16 # number of inputs per batch per device
    num_responses_per_input = 4
    minibatch_size = 8 # for training only
    num_minibatches = batch_size * num_responses_per_input // minibatch_size
    assert(num_minibatches * minibatch_size == batch_size * num_responses_per_input)

    max_new_tokens = 512
    kl_coeff = 100


    ### Initialize ###
    pretrained_model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", use_cache=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id
    pretrained_model.config.pad_token_id = tokenizer.pad_token_id
    pretrained_model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(pretrained_model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0, target_modules=["q_proj", "v_proj"]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train_steps)

    train_dataset = get_dataset(num_samples_train)
    test_dataset = get_dataset(num_samples_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    accelerator = Accelerator(mixed_precision="fp16")
    model, pretrained_model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, pretrained_model, optimizer, train_dataloader, test_dataloader)

    policy = PolicyREINFORCE(model, pretrained_model, tokenizer, get_rewards, num_responses_per_input=num_responses_per_input, max_new_tokens=max_new_tokens, kl_coeff=kl_coeff, device=accelerator.device)

    if accelerator.is_main_process:
        wandb.init(project=project_name, name=run_name)


    ### Train ###
    num_steps = 0
    train_iter = iter(train_dataloader)
    for train_step in tqdm(range(num_train_steps), disable=not accelerator.is_main_process):

        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
        
        batch.update(tokenizer(list(batch["prompt"]), padding=True, padding_side="left", return_tensors="pt").to(accelerator.device))
        
        ### Sample responses ###
        batch = policy.get_train_samples(batch)
        
        ### Update model ###
        train_stats = {} 
        optimizer.zero_grad()
        for minibatch_idx in range(num_minibatches):
            minibatch = {k: v[minibatch_idx * minibatch_size:(minibatch_idx + 1) * minibatch_size] for k, v in batch.items()}
            minibatch_train_stats = policy.get_loss(minibatch)
            train_stats.update(minibatch_train_stats)
            minibatch_loss = minibatch_train_stats["loss"]/num_minibatches
            accelerator.backward(minibatch_loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        scheduler.step()

        ### Logging ###
        if train_step % num_steps_per_train_log == 0:
            rewards = accelerator.gather(batch["rewards"])
            
            sample_stats = get_sample_stats(batch["rewards"])
            sample_stats = accelerator.gather(sample_stats)
            
            train_stats = accelerator.gather(train_stats)            
            
            if accelerator.is_main_process:
                wandb.log({k: v.mean().item() for k, v in sample_stats.items()}, step=train_step)
                wandb.log({k: v.mean().item() for k, v in train_stats.items()}, step=train_step)
                wandb.log({"rewards": rewards.mean().item()}, step=train_step)
                print(f"Loss: {train_stats['loss'].mean().item()}")
                print(f"Rewards: {rewards.mean().item()}")
                print(f"Accuracy: {sample_stats['ratio_correct'].mean().item()}")
        
        if train_step % num_steps_per_test_log == 0:
            model.eval()
            for test_step, test_batch in enumerate(test_dataloader):
                test_batch.update(tokenizer(list(test_batch["prompt"]), padding=True, padding_side="left", return_tensors="pt").to(accelerator.device))
                test_batch = policy.get_test_samples(test_batch)
                
            test_rewards = accelerator.gather(test_batch["rewards"])
            
            test_sample_stats = get_sample_stats(test_batch["rewards"])
            test_sample_stats = accelerator.gather(test_sample_stats)
            
            if accelerator.is_main_process:
                wandb.log({"test "+k: v.mean().item() for k, v in test_sample_stats.items()}, step=train_step)
                wandb.log({"test rewards": test_rewards.mean().item()}, step=train_step)
                print(f"Test accuracy: {test_sample_stats['ratio_correct'].mean().item()}")
                

if __name__ == "__main__":
    main()