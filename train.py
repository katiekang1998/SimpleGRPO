

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import peft
import wandb

from dataset import get_dataset, custom_collate
from policy_reinforce import PolicyREINFORCE

model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# learning_rate = 
# num_train_steps = 
# project_name = 
# run_name = 
num_samples_train = 10000
num_samples_test = 1000
batch_size = 32
device = "cuda:0"

model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.bos_token

# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train_steps)

train_dataset = get_dataset(num_samples_train)
test_dataset = get_dataset(num_samples_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

# model = peft.get_peft_model(model, peft.LoraConfig(r=16, lora_alpha=16, lora_dropout=0, target_modules=["q_proj", "v_proj"]))
# model.to(device)

policy = PolicyREINFORCE(model, tokenizer)

# wandb.init(project=project_name, name=run_naame)




num_steps = 0
for epoch in range(1):

    model.train()
    for batch in train_dataloader:
        batch.update(tokenizer(batch["prompt"], padding=True, padding_side="left", return_tensors="pt").to(device))
        
        loss = policy.get_loss(batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()



