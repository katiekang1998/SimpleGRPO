from reward import get_rewards
import torch.nn as nn
import numpy as np

class PolicyREINFORCE(nn.Module):
    
    def __init__(self, model, tokenizer, num_responses_per_input=4, max_new_tokens=256):
        super(PolicyREINFORCE, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.num_responses_per_input = num_responses_per_input
        self.max_new_tokens = max_new_tokens
    
    def get_loss(self, batch):
        import IPython; IPython.embed()
        
        # 
        samples = self.get_samples(batch, self.num_responses_per_input)
        samples_text = self.tokenizer.batch_decode(samples, skip_special_tokens=True)
        rewards = get_rewards(samples_text, 
                              np.repeat(batch["target"], self.num_responses_per_input), 
                              np.repeat(batch["numbers"], self.num_responses_per_input))
        # rewards is wrong
        
        
        
        values = np.mean(np.reshape(rewards, (-1, self.num_responses_per_input)), axis=1)
        values = np.repeat(values, self.num_responses_per_input)
        
        probs_all = model(samples).logits.softmax(dim=-1)[:, batch["input_ids"].shape[1]-1: -1]
        
        
        
    #     return loss
    
    def get_samples(self, batch, num_responses_per_input=1):
        
        samples = self.model.generate(input_ids = batch["input_ids"], 
                            attention_mask = batch["attention_mask"],
                            max_new_tokens=self.max_new_tokens,
                            do_sample=True, 
                            num_return_sequences=num_responses_per_input)
        return samples
    
    
    



        # generated_outputs = model.generate(questions_ids, max_length=20+questions_ids.shape[1], pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=True, num_return_sequences=4, return_dict_in_generate=True, output_scores=True)

        # generated_sequences = generated_outputs.sequences
        # generated_sequences_text = tokenizer.batch_decode(generated_sequences[:, questions_ids.shape[1]:], skip_special_tokens=True)

        # #repeat each element in answers_text 10 times
        # answers_text = [a for a in answers_text for _ in range(4)]

        # rewards = list(map(get_reward2, generated_sequences_text, answers_text))
        
        
        # values = np.mean(np.reshape(rewards, (-1, 4)), axis=1)
        # values = np.repeat(values, 4)


        # probs_all = model(generated_sequences).logits.softmax(dim=-1)[:, questions_ids.shape[1]-1: -1]
        # log_probs = torch.log(torch.take_along_dim(probs_all, generated_sequences[:, questions_ids.shape[1]:].unsqueeze(-1), dim=2).squeeze().clip(1e-20, 1))
        # mask = 1 - (generated_sequences[:, questions_ids.shape[1]:] ==tokenizer.pad_token_id).int()
        # log_probs = log_probs*mask

        # # loss = -torch.mean(torch.sum(log_probs, dim=-1) * (torch.tensor(rewards, dtype=torch.float32)))

        # loss = -torch.mean(torch.sum(log_probs, dim=-1) * (torch.tensor(rewards - values, dtype=torch.float32)))