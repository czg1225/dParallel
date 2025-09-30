import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
)
from transformers import Trainer, TrainingArguments
from peft import (
    LoraConfig,
    get_peft_model
)
import deepspeed
from typing import Dict, Any
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import types
import random


def load_config(config_path: str) -> Dict[str, Any]:
    """Loading a YAML Configuration File"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_deepspeed_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Creating a DeepSpeed ​​Configuration"""
    return {
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": True,
  "bf16": {
    "enabled": "auto"
  },

  "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
}

def prepare_model(config: Dict[str, Any]):
    """Prepare the model and tokenizer according to the configuration"""

    # Setting torch dtype
    torch_dtype = getattr(torch, config['model']['torch_dtype'])
    
    # Loading the model and tokenizer
    model = AutoModel.from_pretrained(
        config['model']['name'],
        torch_dtype=torch_dtype,
        trust_remote_code=config['model']['trust_remote_code'],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=config['model']['trust_remote_code']
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configuring LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type']
    )

    # Preparing the model for training
    model = get_peft_model(model, lora_config)
    
    # Print the number of trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


def forward_process_semi_ar(input_ids, prompt_lengths, mask_token_id=126336, block_size=32, eps=1e-3):
    """Semi-autoregressive forward masking process"""
    b, l = input_ids.shape
    device = input_ids.device
    
    noisy_batch = input_ids.clone()
    noisy_batch_rev = input_ids.clone()
    masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
    masked_indices_rev = torch.zeros_like(input_ids, dtype=torch.bool)

    # prompt mask
    token_positions = torch.arange(l, device=device).expand(b, l)
    prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
    

    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    noisy_batch_rev[prompt_mask] = input_ids[prompt_mask]
    
    # semi-autoregressive mask
    for i in range(b):
        prompt_len = prompt_lengths[i].item()
        response_len = l - prompt_len
        
        if response_len > 0:

            max_blocks = response_len // block_size
            
            num_blocks_to_mask = random.randint(0, max_blocks)
            num_tokens_to_mask = num_blocks_to_mask * block_size
            
            mask_start = prompt_len + num_tokens_to_mask
            if num_blocks_to_mask == max_blocks:
                mask_end = l
            else:
                mask_end = mask_start + block_size
            
            # t = torch.rand(b, device=input_ids.device)
            t = torch.full((b,), 0.5, device=input_ids.device)  # 50% mask ratio
            p_mask = (1 - eps) * t + eps
            seg_len = mask_end - mask_start
            p_mask = p_mask[:, None].repeat(1, seg_len)
            seg_mask = torch.rand((b, seg_len), device=input_ids.device) < p_mask
            masked_indices[:, mask_start:mask_end] = seg_mask
            masked_indices_rev[:, mask_start:mask_end] = ~seg_mask
            
            noisy_batch = torch.where(masked_indices, 126336, input_ids)
            noisy_batch[i, mask_end:l] = mask_token_id

            noisy_batch_rev = torch.where(masked_indices_rev, 126336, input_ids)
            noisy_batch_rev[i, mask_end:l] = mask_token_id


    return noisy_batch, noisy_batch_rev, masked_indices, masked_indices_rev


class DLMTrainer(Trainer):
    """自定义的扩散语言模型训练器"""
    
    def __init__(self, mask_token_id=126336, **kwargs):
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id
        self.temperature = 0.5
        self.entropy_weight = 2
        self.block_size = 32

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        prompt_lengths = inputs["prompt_lengths"]
    
        # Semi-autoregressive forward masking
        noisy_batch, noisy_batch_rev, masked_indices, masked_indices_rev = forward_process_semi_ar(
            input_ids, prompt_lengths, self.mask_token_id, self.block_size
        )

        # compute logits
        outputs = model(input_ids=noisy_batch)
        logits = outputs.logits

        # compute logits for complementary mask
        outputs_rev = model(input_ids=noisy_batch_rev)
        logits_rev = outputs_rev.logits

        # Calculate loss: only calculate loss for masked tokens
        if masked_indices.sum() > 0:
            # Get the logits and labels of the masked positions
            masked_logits = logits[masked_indices]  # [num_masked, vocab_size]
            masked_labels = input_ids[masked_indices]  # [num_masked]
            
            # cross entropy loss
            token_loss = F.cross_entropy(
                masked_logits, 
                masked_labels, 
                reduction='none'
            ) 

            ce_loss = torch.sum(token_loss) / masked_indices.sum()
        else:
            ce_loss = 0.0 * logits.sum()
        

        # Calculate loss: only calculate loss for masked tokens
        if masked_indices_rev.sum() > 0:
            # Get the logits and labels of the masked positions
            masked_logits_rev = logits_rev[masked_indices_rev]  # [num_masked, vocab_size]
            masked_labels_rev = input_ids[masked_indices_rev]  # [num_masked]
            
            # cross entropy loss
            token_loss_rev = F.cross_entropy(
                masked_logits_rev, 
                masked_labels_rev, 
                reduction='none'
            ) 
            
            ce_loss_rev = torch.sum(token_loss_rev) / masked_indices_rev.sum()
        else:
            ce_loss_rev = 0.0 * logits_rev.sum()


        # ---------- Apply entropy loss only to “correctly predicted” tokens ----------
        if masked_indices.sum() > 0:
            # Calculate the probability and entropy of each position
            # Note: argmax is not affected by temperature; logits/probs are equivalent.
            probs = F.softmax(logits / self.temperature, dim=-1)           # [B, T, V]
            H_tok = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)        # [B, T]

            # predictions
            pred_ids = logits.argmax(dim=-1)                                # [B, T]

            # Only keep: positions that are masked and predicted == label
            correct_mask = (pred_ids == input_ids) & masked_indices  # [B, T] bool

            num_correct = correct_mask.sum()
            if num_correct.item() > 0:
                # Minimize entropy only for the "correctly predicted" positions
                entropy_loss = (H_tok * correct_mask).sum() / num_correct.clamp_min(1)
            else:
                entropy_loss = 0.0 * logits.sum()
        else:
            entropy_loss = 0.0 * logits.sum()
        


        # ---------- Apply entropy loss only to “correctly predicted” tokens ----------
        if masked_indices_rev.sum() > 0:
            # Calculate the probability and entropy of each position
            # Note: argmax is not affected by temperature; logits/probs are equivalent.
            probs_rev = F.softmax(logits_rev / self.temperature, dim=-1)           # [B, T, V]
            H_tok_rev = -(probs_rev * torch.log(probs_rev + 1e-12)).sum(dim=-1)        # [B, T]

            # predictions
            pred_ids_rev = logits_rev.argmax(dim=-1)                                # [B, T]

            # Only keep: positions that are masked and predicted == label
            correct_mask_rev = (pred_ids_rev == input_ids) & masked_indices_rev  # [B, T] bool

            num_correct_rev = correct_mask_rev.sum()
            if num_correct_rev.item() > 0:
                # Minimize entropy only for the "correctly predicted" positions
                entropy_loss_rev = (H_tok_rev * correct_mask_rev).sum() / num_correct_rev.clamp_min(1)
            else:
                entropy_loss_rev = 0.0 * logits_rev.sum()
        else:
            entropy_loss_rev = 0.0 * logits_rev.sum()

        
        # ==================== combined total loss ====================
        total_loss = ce_loss + ce_loss_rev + self.entropy_weight*(entropy_loss + entropy_loss_rev)
        
        return (total_loss, outputs) if return_outputs else total_loss




def main():
    # 1. Loading configuration, model and tokenizer
    config = load_config('configs/config_lora_llada.yaml')
    
    # 2. Setting training parameters
    training_args = TrainingArguments(
        **config['training'],
        deepspeed=get_deepspeed_config(config),
        ddp_find_unused_parameters=False,
        label_names=["input_ids", "prompt_lengths"]
    )
    
    model, tokenizer = prepare_model(config)
    
    # 3. Load the original dataset
    dataset = load_dataset(
        "json", 
        data_files="data/llada_train_data_numi_tail.json", 
        split="train"
    )

    # 4. Format each sample, generate the complete text and record the number of tokens in the prompt section
    def format_example(example):
        texts = []
        prompt_lengths = []
        
        for question, response in zip(example["question"], example["llm_response"]):

            # messages = [{"role": "user", "content": question}]
            # prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
   
            # prompt text
            prompt_text = question
            
            # response text
            answer_text = response + tokenizer.eos_token
            
            # complete text
            full_text = prompt_text + answer_text
            texts.append(full_text)
            
            # Calculate the number of tokens in the prompt part
            prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            prompt_lengths.append(len(prompt_token_ids))
        
        return {"text": texts, "prompt_length": prompt_lengths}

    formatted_dataset = dataset.map(
        format_example,
        batched=True,
    )

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  
            max_length=512,
            add_special_tokens=False,  
        )
        
        tokenized["prompt_lengths"] = examples["prompt_length"]
        
        return tokenized

    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
    )

    from dataclasses import dataclass
    from typing import Dict, List, Any
    import torch

    @dataclass
    class MaskDiffusionDataCollator:
        tokenizer: Any
        pad_to_max_length: bool = False  
        max_length: int = 384  
        
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            input_ids = [torch.tensor(f["input_ids"]) for f in features]
            prompt_lengths = [f["prompt_lengths"] for f in features]
            
            target_length = max_length
            
            pad_token_id = self.tokenizer.eos_token_id
            
            # right padding
            padded_input_ids = []
            for ids in input_ids:
                current_length = len(ids)
                if current_length < target_length:
                    # Right padding with EOS token
                    padding_length = target_length - current_length
                    padded_ids = torch.cat([
                        ids, 
                        torch.full((padding_length,), pad_token_id, dtype=ids.dtype)
                    ])
                else:
                    padded_ids = ids[:target_length]
                
                padded_input_ids.append(padded_ids)
            
            batch = {
                "input_ids": torch.stack(padded_input_ids),
                "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long)
            }
            
            return batch

    data_collator_fixed = MaskDiffusionDataCollator(
        tokenizer=tokenizer,
        pad_to_max_length=True,
        max_length=384
    )

    # 6. 创建DLM训练器
    trainer = DLMTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator_fixed,  # 或者使用 data_collator_fixed
        mask_token_id=126336,  # [MASK] token的ID
    )
       
    # 6. 开始训练
    trainer.train()

if __name__ == "__main__":
    main()