import torch.nn as nn
import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F

class PolicyGRPO(nn.Module):
    
    def __init__(self, model, pretrained_model, tokenizer, reward_fn, num_responses_per_input=4, max_new_tokens=256, kl_coeff=0.1, clip_range=0.01, device="cuda:0"):
        super(PolicyGRPO, self).__init__()
        self.model = model
        self.pretrained_model = pretrained_model
        self.tokenizer = tokenizer
        self.num_responses_per_input = num_responses_per_input
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.reward_fn = reward_fn
        self.kl_coeff = kl_coeff
        self.clip_range = clip_range

        
    def get_train_samples(self, batch):
        # b: batch size
        # s1: input sequence length
        # s2: output sequence length
        # n: num responses per input
        # v: vocab size

        # input_ids: b x s1
        input_ids = batch["input_ids"]

        sample_model = self.model.module if hasattr(self.model, 'module') else self.model
        with autocast():
            with torch.no_grad():
                # samples: bn x (s1+s2)
                input_output_ids = sample_model.generate(input_ids = batch["input_ids"], 
                                    attention_mask = batch["attention_mask"],
                                    max_new_tokens = self.max_new_tokens,
                                    min_new_tokens = self.max_new_tokens,
                                    do_sample = True, 
                                    num_return_sequences = self.num_responses_per_input,
                                    pad_token_id = self.tokenizer.pad_token_id)
                                    # return_dict_in_generate=True)
                                    # output_scores=True,)
                
        # output_ids: bn x s2
        output_ids = input_output_ids[:, input_ids.shape[1]:]
        
        with autocast():
            with torch.no_grad():
                # # logits: mb x s1+s2 x v
                old_logits = self.model(input_output_ids).logits
        # # logits: mb x s2 x v
        old_logits = old_logits[:, input_ids.shape[1]-1: -1]
        old_log_probs = F.log_softmax(old_logits, dim=-1)
        old_log_probs = torch.gather(old_log_probs, 2, output_ids.unsqueeze(-1)).squeeze(-1)
        mask = (output_ids != self.tokenizer.pad_token_id).int()
        old_log_probs *= mask
        old_log_probs = old_log_probs.detach()
        

        # # logits: bn x s2 x v
        # old_logits = torch.stack(sample_dict.scores, dim=1)
        # old_probs = F.softmax(old_logits, -1)
        # old_probs = torch.gather(old_probs, -1, output_ids.unsqueeze(-1)).squeeze(-1)
        # old_probs = torch.clamp(old_probs, min=1e-9)
        # old_log_probs = old_probs.log()
        # mask = (output_ids != self.tokenizer.pad_token_id).int()
        # old_log_probs *= mask
        # old_log_probs = old_log_probs.detach()

        # samples_text: bn
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # rewards: bn
        rewards = self.reward_fn(output_text, 
                                 np.repeat(batch["target"], self.num_responses_per_input), 
                                 np.repeat(batch["numbers"], self.num_responses_per_input, axis=0))
        
        
        # values: b
        values = np.mean(np.reshape(rewards, (-1, self.num_responses_per_input)), axis=1)
        # vales: bn
        values = np.repeat(values, self.num_responses_per_input)

        with autocast():
            with torch.no_grad():
                # logits: bn x s1+s2 x v
                pretrained_logits = self.pretrained_model(input_output_ids).logits
                # logits: bn x s2 x v
                pretrained_logits = pretrained_logits[:, input_ids.shape[1]-1: -1]

                pretrined_log_probs = F.log_softmax(pretrained_logits, dim=-1)
                # pretrined_probs: bn x s2
                pretrined_log_probs = torch.gather(pretrined_log_probs, 2, output_ids.unsqueeze(-1)).squeeze(-1)
                mask = (output_ids != self.tokenizer.pad_token_id).int()
                pretrined_log_probs *= mask
                pretrined_log_probs = pretrined_log_probs.detach()
        

        output_batch = {}
        output_batch["input_ids"] = torch.repeat_interleave(input_ids, self.num_responses_per_input, 0)
        output_batch["input_output_ids"] = input_output_ids
        output_batch["rewards"] = torch.Tensor(rewards).to(self.device)
        output_batch["values"] = torch.Tensor(values).to(self.device)
        output_batch["pretrained_log_probs"] = pretrined_log_probs
        output_batch["old_log_probs"] = old_log_probs

        return output_batch
    
    
    def get_test_samples(self, batch):
        input_ids = batch["input_ids"]

        sample_model = self.model.module if hasattr(self.model, 'module') else self.model
        with autocast():
            with torch.no_grad():
                # samples: b x (s1+s2)
                input_output_ids = sample_model.generate(input_ids = batch["input_ids"], 
                                    attention_mask = batch["attention_mask"],
                                    max_new_tokens=self.max_new_tokens,
                                    do_sample=True, 
                                    pad_token_id=self.tokenizer.pad_token_id)
        # output_ids: b x s2
        output_ids = input_output_ids[:, input_ids.shape[1]:]

        # samples_text: b
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # rewards: b
        rewards = self.reward_fn(output_text, 
                                 batch["target"], 
                                 batch["numbers"])
        
        batch["input_output_ids"] = input_output_ids
        batch["rewards"] = torch.Tensor(rewards).to(self.device)
        return batch
    
    
    def get_loss(self, batch): 
        # mb: minibatch size
        # s1: input sequence length
        # s2: output sequence length
        # v: vocab size
        
        input_output_ids = batch["input_output_ids"]
        input_ids = batch["input_ids"]
        output_ids = input_output_ids[:, input_ids.shape[1]:]
        
        # logits: mb x s1+s2 x v
        logits = self.model(input_output_ids).logits
        # logits: mb x s2 x v
        logits = logits[:, input_ids.shape[1]-1: -1]
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = torch.gather(log_probs, 2, output_ids.unsqueeze(-1)).squeeze(-1)
        mask = (output_ids != self.tokenizer.pad_token_id).int()
        log_probs = log_probs * mask

        pg_ratio = torch.exp(log_probs.sum(axis=1) - batch["old_log_probs"].sum(axis=1))
        clipped_pg_ratio = torch.clip(pg_ratio, 1-self.clip_range, 1+self.clip_range)
        advantage = (batch["rewards"] - batch["values"])
        pg_loss = -torch.minimum(pg_ratio*advantage, clipped_pg_ratio*advantage).mean()
        clip_ratio = (torch.lt(clipped_pg_ratio*advantage, pg_ratio*advantage)).float().mean()

        kl_log_ratio = batch["pretrained_log_probs"].sum(axis=1) - log_probs.sum(axis=1)
        kl_loss = torch.mean(torch.exp(kl_log_ratio) - kl_log_ratio - 1)
        
        loss = pg_loss + self.kl_coeff*kl_loss
        
        print("pg_ratio ", pg_ratio)
        
        return {"loss": loss, "pg_loss": pg_loss, "kl_loss": self.kl_coeff*kl_loss, "clip_ratio": clip_ratio}