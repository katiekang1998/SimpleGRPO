




class PolicyGRPO(nn.Module):
    
    def __init__(self, model, cliprange):
        self.model_reference = model 
        self.model_current = copy.deepcopy(model)
        self.cliprange = cliprange
        
    
    # def get_advantage(self):
    #     return advantage 
    
    # def get_kl(self):
    #     return kl 
    
    def get_loss(self, batch_inputs):
        
        # rewards = get_rewards(samples)
        return loss
    
    # def get_samples(self, batch_inputs, num_responses_per_input=1):
        
        
    #     self.model.generate(questions_ids, 
    #                         max_length=max_new_tokens, 
    #                         pad_token_id=tokenizer.pad_token_id, 
    #                         eos_token_id=tokenizer.eos_token_id, 
    #                         do_sample=True, 
    #                         num_return_sequences=num_responses_per_input)
    #                         # return_dict_in_generate=True, 
    #                         # output_scores=True)
    #     return samples