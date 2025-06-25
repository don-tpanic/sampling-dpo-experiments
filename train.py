import os
from dataclasses import dataclass
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from typing import Tuple, Dict, List, Any
from utils import forward_pass_compute_logprobs_n_selfcertainties as forward_logprobs

"""
Training script for DPO
"""

@dataclass
class TrainConfig:
    lr: float = 3e-5
    epochs: int = 2
    batch_size: int = 4
    device: str = 'cuda'
    validation_split: float = 0.1
    max_absolute_length: int = 2048
    beta: float = 0.1


def create_train_val_split(dataset: Dataset, val_split: float = 0.1) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into training and validation sets.
    """
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    indices = list(range(total_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    return train_dataset, val_dataset


def prepare_dpo_sample(
        question: str,
        chosen_solution: str,
        reject_solution: str,
        tokenizer: AutoTokenizer,
        max_absolute_length: int = 2048
    ) -> Dict[str, torch.Tensor]:
    """
    Prepare a single DPO training sample by creating full sequences
    (question + answer) for both chosen and rejected responses.
    No truncation unless absolutely necessary for memory constraints.
    """
    # Create chat templates for chosen and rejected responses
    chosen_chat = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": chosen_solution}
    ]
    
    reject_chat = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reject_solution}
    ]
    
    # Apply chat template and tokenize
    chosen_text = tokenizer.apply_chat_template(
        chosen_chat, tokenize=False, add_generation_prompt=False
    )
    reject_text = tokenizer.apply_chat_template(
        reject_chat, tokenize=False, add_generation_prompt=False
    )

    # Tokenize the full sequences without truncation (except safety limit)
    chosen_tokens = tokenizer(
        chosen_text,
        truncation=True,
        max_length=max_absolute_length,
        return_tensors="pt",
        padding=False
    )
    
    reject_tokens = tokenizer(
        reject_text,
        truncation=True,
        max_length=max_absolute_length,
        return_tensors="pt",
        padding=False
    )
    
    # Also tokenize just the question part to identify where the answer starts
    question_chat = [{"role": "user", "content": question}]
    question_text = tokenizer.apply_chat_template(
        question_chat, tokenize=False, add_generation_prompt=True
    )
    question_tokens = tokenizer(
        question_text,
        truncation=True,
        max_length=max_absolute_length,
        return_tensors="pt",
        padding=False
    )

    # Calculate answer start positions and lengths for both chosen and rejected solutions
    chosen_answer_start = len(question_tokens["input_ids"][0])
    reject_answer_start = len(question_tokens["input_ids"][0])
    chosen_answer_length = len(chosen_tokens["input_ids"][0]) - chosen_answer_start
    reject_answer_length = len(reject_tokens["input_ids"][0]) - reject_answer_start
    
    return {
        "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
        "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
        "reject_input_ids": reject_tokens["input_ids"].squeeze(0),
        "reject_attention_mask": reject_tokens["attention_mask"].squeeze(0),
        "chosen_answer_start": torch.tensor(chosen_answer_start, dtype=torch.long),
        "chosen_answer_length": torch.tensor(chosen_answer_length, dtype=torch.long),
        "reject_answer_start": torch.tensor(reject_answer_start, dtype=torch.long),
        "reject_answer_length": torch.tensor(reject_answer_length, dtype=torch.long),
    }


def collate_fn(
        batch: List[Dict[str, Any]], 
        tokenizer: AutoTokenizer, 
        max_absolute_length: int = 2048
    ) -> Dict[str, torch.Tensor]:
    """
    Collate function for DPO training with dynamic batch-level padding.
    Each batch is padded to the maximum length within that batch.
    """
    # Prepare all samples
    prepared_samples = []
    for item in batch:
        sample = prepare_dpo_sample(
            question=item["questions"],
            chosen_solution=item["chosen_solutions"],
            reject_solution=item["reject_solutions"],
            tokenizer=tokenizer,
            max_absolute_length=max_absolute_length
        )
        # Add the logprobs produced by the initial (reference) model
        sample["chosen_logprobs"] = item["chosen_logprobs"]
        sample["reject_logprobs"] = item["reject_logprobs"]
        prepared_samples.append(sample)
    
    # Extract all tensors for batching
    chosen_input_ids = [sample["chosen_input_ids"] for sample in prepared_samples]
    chosen_attention_masks = [sample["chosen_attention_mask"] for sample in prepared_samples]
    chosen_answer_starts = [sample["chosen_answer_start"] for sample in prepared_samples]
    chosen_answer_lengths = [sample["chosen_answer_length"] for sample in prepared_samples]
    reject_input_ids = [sample["reject_input_ids"] for sample in prepared_samples]
    reject_attention_masks = [sample["reject_attention_mask"] for sample in prepared_samples]
    reject_answer_starts = [sample["reject_answer_start"] for sample in prepared_samples]
    reject_answer_lengths = [sample["reject_answer_length"] for sample in prepared_samples]

    # Find the maximum length in this batch for dynamic padding
    max_chosen_length = max(seq.shape[0] for seq in chosen_input_ids)
    max_reject_length = max(seq.shape[0] for seq in reject_input_ids)
    batch_max_length = max(max_chosen_length, max_reject_length)
    
    # Pad sequences to the maximum length within this specific batch
    def pad_to_length(tensors, target_length, pad_value):
        padded = []
        for tensor in tensors:
            current_length = tensor.shape[0]
            if current_length < target_length:
                padding = torch.full((target_length - current_length,), pad_value, dtype=tensor.dtype)
                padded_tensor = torch.cat([tensor, padding], dim=0)
            else:
                padded_tensor = tensor
            padded.append(padded_tensor)
        return torch.stack(padded)
    
    chosen_input_ids = pad_to_length(chosen_input_ids, batch_max_length, tokenizer.pad_token_id)
    chosen_attention_masks = pad_to_length(chosen_attention_masks, batch_max_length, 0)
    reject_input_ids = pad_to_length(reject_input_ids, batch_max_length, tokenizer.pad_token_id)
    reject_attention_masks = pad_to_length(reject_attention_masks, batch_max_length, 0)
    
    # Collect other data
    chosen_logprobs = torch.stack([torch.tensor(sample["chosen_logprobs"]) for sample in prepared_samples])
    reject_logprobs = torch.stack([torch.tensor(sample["reject_logprobs"]) for sample in prepared_samples])

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_masks,
        "chosen_answer_start": torch.stack(chosen_answer_starts),
        "chosen_answer_length": torch.stack(chosen_answer_lengths),
        "reject_answer_start": torch.stack(reject_answer_starts),
        "reject_answer_length": torch.stack(reject_answer_lengths),
        "reject_input_ids": reject_input_ids,
        "reject_attention_mask": reject_attention_masks,
        "chosen_logprobs": chosen_logprobs,
        "reject_logprobs": reject_logprobs,
        "batch_max_length": batch_max_length,
    }


def compute_dpo_loss(
        model_chosen_logprobs: torch.Tensor,
        model_rejected_logprobs: torch.Tensor,
        reference_chosen_logprobs: torch.Tensor,
        reference_rejected_logprobs: torch.Tensor,
        beta: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        A tuple of three tensors: (loss, chosen_rewards, rejected_rewards)
    """
    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios
    
    # DPO loss
    losses = -F.logsigmoid(logits * beta)

    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    return losses.mean(), chosen_rewards, rejected_rewards


def compute_dpo_loss_batch(
        batch,
        policy_model,
        tokenizer,
        device,
        beta
    ):
    policy_chosen_log_probas, _ = forward_logprobs(
        model=policy_model,
        tokenizer=tokenizer,
        input_ids=batch["chosen_input_ids"].to(device),
        attention_mask=batch["chosen_attention_mask"].to(device),
        answer_start_positions=batch["chosen_answer_start"].to(device),
        answer_lengths=batch["chosen_answer_length"].to(device),
        no_grad=False,
    )
    policy_rejected_log_probas, _ = forward_logprobs(
        model=policy_model,
        tokenizer=tokenizer,
        input_ids=batch["reject_input_ids"].to(device),
        attention_mask=batch["reject_attention_mask"].to(device),
        answer_start_positions=batch["reject_answer_start"].to(device),
        answer_lengths=batch["reject_answer_length"].to(device),
        no_grad=False,
    )

    reference_model_log_probas = batch["chosen_logprobs"].to(device)
    reference_rejected_log_probas = batch["reject_logprobs"].to(device)

    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas,
        model_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=reference_model_log_probas,
        reference_rejected_logprobs=reference_rejected_log_probas,
        beta=beta
    )
    return loss, chosen_rewards, rejected_rewards


def train(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        config: TrainConfig,
    ) -> AutoModelForCausalLM:
    """
    Main training function for DPO.
    """
    train_dataset, val_dataset = create_train_val_split(dataset, config.validation_split)
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Create collate function with tokenizer
    def batch_collate_fn(batch):
        return collate_fn(batch, tokenizer, config.max_absolute_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=batch_collate_fn,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=batch_collate_fn,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr)
    
    # Move model to device
    model = model.to(config.device)
    
    # Training loop
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            # Compute loss and rewards
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=model,
                tokenizer=tokenizer,
                device=config.device,
                beta=config.beta
            )
            
            loss.backward()
            import code; code.interact(local=locals())  # Debugging breakpoint
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / num_train_batches
        print(f"  Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        pass # TODO:
    
    return model

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    prepared_training_data = torch.load("data/prepared_training_data_temp1.5_bsz8_shot2_q500.pt")
    hf_dataset = Dataset.from_dict({
        "questions": prepared_training_data["questions"],
        "chosen_solutions": prepared_training_data["chosen_solutions"],
        "chosen_logprobs": prepared_training_data["chosen_logprobs"],
        "chosen_selfcertainties": prepared_training_data["chosen_selfcertainties"],
        "reject_solutions": prepared_training_data["reject_solutions"],
        "reject_logprobs": prepared_training_data["reject_logprobs"],
        "reject_selfcertainties": prepared_training_data["reject_selfcertainties"],
    })
    train(model, tokenizer, hf_dataset, TrainConfig())