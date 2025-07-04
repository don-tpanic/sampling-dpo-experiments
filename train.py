import argparse
import os
from dataclasses import dataclass
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
import wandb
import tqdm
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from datasets import Dataset
from typing import Tuple, Dict, List, Any
import copy
import utils
from utils import forward_pass_compute_logprobs_n_selfcertainties as forward_logprobs

"""
Training script for DPO
"""

@dataclass
class TrainConfig:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
    

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
        {
            "role": "user", 
            "content": prompt_template.format(
                n_shot_examples=n_shot_examples,
                question=question
            )
        },
        {"role": "assistant", "content": chosen_solution}
    ]
    
    reject_chat = [
        {
            "role": "user", 
            "content": prompt_template.format(
                n_shot_examples=n_shot_examples,
                question=question
            )
        },
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
    # No padding as this is a single sample.
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
    question_chat = [
        {
            "role": "user", 
            "content": prompt_template.format(
                n_shot_examples=n_shot_examples,
                question=question
            )
        }
    ]
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

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_masks,
        "chosen_answer_start": torch.stack(chosen_answer_starts),
        "chosen_answer_length": torch.stack(chosen_answer_lengths),
        "reject_answer_start": torch.stack(reject_answer_starts),
        "reject_answer_length": torch.stack(reject_answer_lengths),
        "reject_input_ids": reject_input_ids,
        "reject_attention_mask": reject_attention_masks,
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
        reference_model,
        tokenizer,
        device,
        beta,
        no_grad,
    ):
    policy_chosen_log_probas, _ = forward_logprobs(
        model=policy_model,
        tokenizer=tokenizer,
        input_ids=batch["chosen_input_ids"].to(device),
        attention_mask=batch["chosen_attention_mask"].to(device),
        answer_start_positions=batch["chosen_answer_start"].to(device),
        answer_lengths=batch["chosen_answer_length"].to(device),
        no_grad=no_grad,
    )
    policy_rejected_log_probas, _ = forward_logprobs(
        model=policy_model,
        tokenizer=tokenizer,
        input_ids=batch["reject_input_ids"].to(device),
        attention_mask=batch["reject_attention_mask"].to(device),
        answer_start_positions=batch["reject_answer_start"].to(device),
        answer_lengths=batch["reject_answer_length"].to(device),
        no_grad=no_grad,
    )

    # Generate reference logprobs on-the-fly using the reference model
    reference_chosen_log_probas, _ = forward_logprobs(
        model=reference_model,
        tokenizer=tokenizer,
        input_ids=batch["chosen_input_ids"].to(device),
        attention_mask=batch["chosen_attention_mask"].to(device),
        answer_start_positions=batch["chosen_answer_start"].to(device),
        answer_lengths=batch["chosen_answer_length"].to(device),
        no_grad=True,
    )
    reference_rejected_log_probas, _ = forward_logprobs(
        model=reference_model,
        tokenizer=tokenizer,
        input_ids=batch["reject_input_ids"].to(device),
        attention_mask=batch["reject_attention_mask"].to(device),
        answer_start_positions=batch["reject_answer_start"].to(device),
        answer_lengths=batch["reject_answer_length"].to(device),
        no_grad=True,
    )

    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas,
        model_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=reference_chosen_log_probas,
        reference_rejected_logprobs=reference_rejected_log_probas,
        beta=beta
    )
    return loss, chosen_rewards, rejected_rewards


def setup_lora_model(model: AutoModelForCausalLM, config: TrainConfig) -> AutoModelForCausalLM:
    """
    Setup LoRA configuration and wrap the model if LoRA is enabled.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    if not getattr(config, 'use_lora', False):
        return model
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=getattr(config, 'lora_r', 16),
        lora_alpha=getattr(config, 'lora_alpha', 32),
        lora_dropout=getattr(config, 'lora_dropout', 0.1),
        target_modules=getattr(config, 'lora_target_modules', ["q_proj", "v_proj"]),
        bias=getattr(config, 'lora_bias', "none"),
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        config: TrainConfig,
    ) -> AutoModelForCausalLM:
    """
    Main training function for DPO.
    """
    run_id = wandb.run.id
    print(f"Run ID: {run_id}")

    model_dir = f"models/{args.config}/{run_id}"
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model will be saved to: {model_dir}")

    # Create reference model as a frozen copy before applying LoRA
    reference_model = copy.deepcopy(model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # Setup LoRA if enabled
    model = setup_lora_model(model, config)

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
    reference_model = reference_model.to(config.device)
    
    # Get gradient accumulation steps (default to 1 if not specified)
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    effective_batch_size = config.batch_size * gradient_accumulation_steps
    
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    
    # Training loop
    for epoch in tqdm.tqdm(range(config.epochs), desc="Training"):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        total_train_chosen_rewards = 0
        total_train_rejected_rewards = 0
        num_train_batches = 0
        accumulation_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            # Compute loss and rewards
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=model,
                reference_model=reference_model,
                tokenizer=tokenizer,
                device=config.device,
                beta=config.beta,
                no_grad=False
            )
            
            loss = loss / gradient_accumulation_steps
            loss.backward()
            accumulation_loss += loss.item()
            
            # Only step optimizer and zero gradients every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # Log the accumulated loss (multiply back by gradient_accumulation_steps for original scale)
                actual_loss = accumulation_loss * gradient_accumulation_steps
                
                # Accumulate metrics
                total_train_loss += actual_loss
                total_train_chosen_rewards += chosen_rewards.mean().item()
                total_train_rejected_rewards += rejected_rewards.mean().item()
                num_train_batches += 1
                
                # Log metrics every 10 accumulated steps
                if (batch_idx + 1) // gradient_accumulation_steps % 10 == 0:
                    batch_metrics = {
                        "batch_loss": actual_loss,
                        "batch_rewards_margin": (chosen_rewards - rejected_rewards).mean().item(),
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                        "accumulated_step": (batch_idx + 1) // gradient_accumulation_steps,
                        "global_step": epoch * (len(train_dataloader) // gradient_accumulation_steps) + ((batch_idx + 1) // gradient_accumulation_steps)
                    }
                    wandb.log(batch_metrics)
                    print(f"  Step {(batch_idx + 1) // gradient_accumulation_steps}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {actual_loss:.4f}")
                
                # Reset accumulation loss
                accumulation_loss = 0
        
        # Handle remaining gradients if the last batch doesn't complete a full accumulation cycle
        if len(train_dataloader) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            
            # Log the remaining accumulated loss
            if accumulation_loss > 0:
                actual_loss = accumulation_loss * gradient_accumulation_steps
                total_train_loss += actual_loss
                num_train_batches += 1
        
        avg_train_loss = total_train_loss / max(num_train_batches, 1)
        avg_train_chosen_rewards = total_train_chosen_rewards / max(num_train_batches, 1)
        avg_train_rejected_rewards = total_train_rejected_rewards / max(num_train_batches, 1)
        print(f"  Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        total_val_chosen_rewards = 0
        total_val_rejected_rewards = 0
        num_val_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                    batch=batch,
                    policy_model=model,
                    reference_model=reference_model,
                    tokenizer=tokenizer,
                    device=config.device,
                    beta=config.beta,
                    no_grad=True
                )
                
                total_val_loss += loss.item()
                total_val_chosen_rewards += chosen_rewards.mean().item()
                total_val_rejected_rewards += rejected_rewards.mean().item()
                num_val_batches += 1
                        
        avg_val_loss = total_val_loss / num_val_batches
        avg_val_chosen_rewards = total_val_chosen_rewards / num_val_batches
        avg_val_rejected_rewards = total_val_rejected_rewards / num_val_batches
        print(f"  Average validation loss: {avg_val_loss:.4f}")

        # Log epoch metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "avg_train_loss": avg_train_loss,
            "avg_train_rewards_margin": avg_train_chosen_rewards - avg_train_rejected_rewards,
            "avg_val_loss": avg_val_loss,
            "avg_val_rewards_margin": avg_val_chosen_rewards - avg_val_rejected_rewards,
            "effective_batch_size": effective_batch_size
        }
        wandb.log(epoch_metrics)

        # Save model checkpoint after each epoch
        model.save_pretrained(f"{model_dir}/checkpoint_epoch_{epoch + 1}")

    # Save final model
    model.save_pretrained(f"{model_dir}/final_model_checkpoint")
    tokenizer.save_pretrained(f"{model_dir}/final_model_checkpoint")
    print("Training complete. Model and tokenizer saved.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training Script")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config_1",
        help="Config filename"
    )
    args = parser.parse_args()
    
    # Load config
    config_dict = utils.load_config(args.config)
    config = TrainConfig(config_dict)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_device
    
    # Load original dataset to get n-shot examples
    dataset = datasets.load_dataset("openai/gsm8k", "main")['train']
    _, _, n_shot_examples = utils.generate_n_shot_examples(
        dataset['question'], dataset['answer'], num_examples=config.num_examples,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    prepared_training_data = torch.load(config.data_path)
    hf_dataset = Dataset.from_dict({
        "questions": prepared_training_data["questions"],
        "chosen_solutions": prepared_training_data["chosen_solutions"],
        "reject_solutions": prepared_training_data["reject_solutions"],
    })

    # Prepare prompt template
    prompt_template = """You are given a math question. You must provide a concise step-by-step reasoning
        and a final answer. Your response should follow strictly the format of the provided examples where each new line is a reasoning step
        written in a very concise style, and the final answer is on the last line. There should be roughly 2-4 steps, but it is okay
        to have more or less steps if needed.
        
        {n_shot_examples}

        # Question:
        {question}
    """

    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)

    # Add LoRA config to wandb logging
    wandb_config = {
        "learning_rate": config.lr,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "device": config.device,
        "validation_split": config.validation_split,
        "max_absolute_length": config.max_absolute_length,
        "beta": config.beta
    }
    
    # Add LoRA parameters to wandb config if enabled
    if getattr(config, 'use_lora', False):
        wandb_config.update({
            "use_lora": True,
            "lora_r": getattr(config, 'lora_r', 16),
            "lora_alpha": getattr(config, 'lora_alpha', 32),
            "lora_dropout": getattr(config, 'lora_dropout', 0.1),
            "lora_target_modules": getattr(config, 'lora_target_modules', ["q_proj", "v_proj"]),
            "lora_bias": getattr(config, 'lora_bias', "none"),
        })

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        config=wandb_config
    )

    train(model, tokenizer, hf_dataset, config)