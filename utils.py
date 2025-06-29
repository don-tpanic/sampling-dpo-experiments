import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import tqdm
from typing import Tuple, List
import yaml


def load_config(config_name):
    """
    Load a YAML configuration file and return the configuration as a dictionary.
    
    Args:
        config_name (str): The name of the configuration file to load.
        
    Returns:
        config_dict: The loaded configuration.
    """
    with open(f"configs/{config_name}.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def generate_n_shot_examples(
        questions: list,
        answers: list,
        num_examples: int = 8,
    ) -> tuple:
    """
    Randomly select `num_examples` from the provided questions and answers,
    and return formatted examples for the prompt and return the remaining questions and answers.
    """
    torch.manual_seed(1234) # Use the same data as examples
    indices = torch.randperm(len(questions))[:num_examples]
    n_shot_examples = []
    for idx, q_idx in enumerate(indices):
        question = questions[q_idx]
        answer = answers[q_idx]
        n_shot_examples.append(
            f"\n# Question:\n{question}"
            f"\n# Answer:\n{answer}"
        )
    n_shot_examples_str = "\n".join(n_shot_examples)
    remaining_questions = [questions[i] for i in range(len(questions)) if i not in indices]
    remaining_answers = [answers[i] for i in range(len(answers)) if i not in indices]
    return remaining_questions, remaining_answers, n_shot_examples_str


def is_correct_solution(
        ans_pred: str, ans_true: str
    ) -> bool:
    """
    Get the numerical value from both `ans_pred` and `ans_true`
    by regex matching `#### <numerical value>`

    # TODO: will need more robust parsing in case the model answers the correct answer but 
    in not an unexpected format.
    """
    ans_pred = ans_pred.split("####")[-1].strip()
    ans_true = ans_true.split("####")[-1].strip()
    try:
        ans_pred = float(ans_pred)
        ans_true = float(ans_true)
    except ValueError:
        return False
    return ans_pred == ans_true


def evaluate_model(
        model: AutoModelForCausalLM,
        eval_dataset: datasets.Dataset,
        tokenizer: AutoTokenizer,
        prompt_template: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        batch_size: int = 8,
        n_shot_examples: str = "",
    ) -> float:
    questions = eval_dataset['question']
    answers = eval_dataset['answer']
    all_scores = []
    
    # Process data in batches
    for batch_start in tqdm.tqdm(range(0, len(questions), batch_size), desc="Evaluating model"):
        batch_end = min(batch_start + batch_size, len(questions))
        batch_questions = questions[batch_start:batch_end]
        batch_answers = answers[batch_start:batch_end]
        
        # Prepare batch of formatted chats
        batch_chats = []
        for question in batch_questions:
            chat = [
                {
                    "role": "user",
                    "content": prompt_template.format(question=question, n_shot_examples=n_shot_examples)
                }
            ]
            formatted_chat = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            batch_chats.append(formatted_chat)
        
        # Tokenize batch with padding
        batch_inputs = tokenizer(
            batch_chats,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = batch_inputs.input_ids.to("cuda")
        attention_mask = batch_inputs.attention_mask.to("cuda")
        
        # Generate responses for the batch
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode batch outputs
        batch_predictions = []
        for i in range(len(batch_questions)):
            # Extract only the generated portion (skip input tokens)
            generated_tokens = output_ids[i][len(input_ids[i]):]
            ans_pred = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            print(f"\n{i}: Decoded answer:\n{ans_pred}")
            batch_predictions.append(ans_pred)
        
        # Evaluate correctness for each item in the batch
        batch_scores = []
        for pred, answer in zip(batch_predictions, batch_answers):
            if is_correct_solution(pred, answer):
                batch_scores.append(1.0)
            else:
                batch_scores.append(0.0)
        
        # Add batch scores to global scores
        all_scores.extend(batch_scores)
    
    # Calculate final accuracy
    return sum(all_scores) / len(all_scores)


def compute_logprobs_n_selfcertainties(
        logits: torch.Tensor,
        gen_ids: torch.Tensor,
        tokenizer: AutoTokenizer,
    ) -> Tuple[List, List]:
    """
    Compute log probabilities and self-certainties for ONLY generated tokens.

    Returns a list of log probabilities and self-certainties for each sample in the batch.
    Padding tokens are ignored in the calculations.

    Args:
        logits (torch.Tensor): A tensor of shape (batch_size, num_new_tokens, vocab_size)
                               containing the logits from the model for generated tokens.
        gen_ids (torch.Tensor): A tensor of shape (batch_size, num_new_tokens) containing
                                  the output token IDs generated by the model.
        tokenizer (AutoTokenizer): The tokenizer used to process the input_ids.
        
    Returns:
        tuple: A tuple containing two lists:
            - log_probs: A list of log probability values for each sample in the batch.
            - self_certainty: A list of self-certainty values for each sample in the batch.
    """
    with torch.no_grad():
        logits = logits.contiguous()
        labels = gen_ids.contiguous()
        batch_size, num_new_tokens, vocab_size = logits.size()

        # Create a mask to ignore padding tokens
        mask = (labels != tokenizer.pad_token_id).float()  # (bsz, num_new_tokens)
        logits = logits.view(batch_size, -1, vocab_size)
        labels = labels.view(batch_size, -1)  # (bsz, num_new_tokens)
        log_probs = F.log_softmax(logits, dim=-1)
        true_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (bsz, num_new_tokens)

        # Apply mask to ignore padding tokens
        true_log_probs = true_log_probs * mask
        # Average over non-padding tokens
        true_log_probs = true_log_probs.sum(dim=-1) / mask.sum(dim=-1)
        assert true_log_probs.shape[0] == gen_ids.shape[0], "Log probabilities shape does not match gen_ids batch size"

        # Compute self-certainty
        uniform_probs = F.softmax(torch.ones_like(logits) / vocab_size, dim=-1)
        kl = F.kl_div(log_probs, uniform_probs, reduction='none')  # (bsz, num_new_tokens, vocab_size)
        # Token-level kl is the sum across vocab size for each token
        kl = kl.sum(dim=-1)  # (bsz, num_new_tokens)
        # Apply mask to ignore padding tokens
        kl = kl * mask
        # Self-certainty is the average KL over tokens in a sequence
        self_certainty = kl.sum(dim=-1) / mask.sum(dim=-1)
        assert self_certainty.shape[0] == gen_ids.shape[0], "Self-certainty shape does not match gen_ids batch size"

    return true_log_probs.tolist(), self_certainty.tolist()


def forward_pass_compute_logprobs_n_selfcertainties(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        answer_start_positions: list,
        answer_lengths: list,
        no_grad: bool = True,
    ) -> Tuple[List, List]:
    """
    Compute log probabilities and self-certainties for ONLY the answer tokens.

    Returns a list of log probabilities and self-certainties for each sample in the batch.
    Padding tokens are ignored in the calculations.

    Args:
        model (AutoModelForCausalLM): The model to compute logits from.
        tokenizer (AutoTokenizer): The tokenizer used to process the input_ids.
        input_ids (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing
                                  the input token IDs which are question+answer.
        attention_mask (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing
                                       the attention mask for the input_ids.
        answer_start_positions (list): A list of integers indicating the start positions of the answers
                                       in the input_ids for each sample in the batch.
        answer_lengths (list): A list of integers indicating the lengths of the answers in the input_ids
                               for each sample in the batch.
        no_grad (bool): If True, disables gradient computation for the forward pass.
        
    Returns:
        tuple: A tuple containing two lists:
            - log_probs: A list of log probability values for each sample in the batch.
            - self_certainty: A list of self-certainty values for each sample in the batch.
    """
    if no_grad is True:
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            logits = outputs.logits
    else:
        model.train()
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        logits = outputs.logits

    # Create labels by shifting input_ids (standard next-token prediction)
    labels = input_ids[:, 1:].contiguous()  # (bsz, seq_len - 1)
    logits = logits[:, :-1, :].contiguous()  # (bsz, seq_len - 1, vocab_size)
    
    batch_size, seq_len, vocab_size = logits.shape
    
    # Create comprehensive mask: ignore prompt tokens AND padding tokens
    mask = torch.zeros_like(labels, dtype=torch.float)  # (bsz, seq_len)
    
    for i in range(batch_size):
        answer_start = answer_start_positions[i]
        answer_length = answer_lengths[i]
    
        # Since labels = input_ids[:, 1:], we need to shift answer_start by -1
        shifted_answer_start = max(0, answer_start - 1)
        shifted_answer_end = min(shifted_answer_start + answer_length, seq_len)
        
        if shifted_answer_start < seq_len and answer_length > 0:
            mask[i, shifted_answer_start:shifted_answer_end] = 1.0
    
    # Also mask out padding tokens
    mask = mask * (labels != tokenizer.pad_token_id).float()
    
    # Reshape for batch computation
    logits = logits.view(batch_size, seq_len, vocab_size)
    labels = labels.view(batch_size, seq_len)
    
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    true_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (bsz, seq_len)
    
    # Apply mask to ignore prompt and padding tokens
    true_log_probs = true_log_probs * mask
    
    # Average over non-masked tokens
    true_log_probs = true_log_probs.sum(dim=-1) / mask.sum(dim=-1)
    assert true_log_probs.shape[0] == input_ids.shape[0], "Log probabilities shape does not match input_ids batch size"

    # Compute self-certainty
    uniform_probs = F.softmax(torch.ones_like(logits) / vocab_size, dim=-1)
    kl = F.kl_div(log_probs, uniform_probs, reduction='none')  # (bsz, seq_len, vocab_size)
    kl = kl.sum(dim=-1)  # (bsz, seq_len)
    kl = kl * mask  # Apply mask to ignore prompt and padding tokens
    self_certainty = kl.sum(dim=-1) / mask.sum(dim=-1)
    assert self_certainty.shape[0] == input_ids.shape[0], "Self-certainty shape does not match input_ids batch size"

    return true_log_probs, self_certainty