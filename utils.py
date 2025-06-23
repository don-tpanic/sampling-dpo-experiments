import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import tqdm


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
        num_examples: int = 8,
    ) -> float:
    questions = eval_dataset['question']
    answers = eval_dataset['answer']
    all_scores = []

    # Generate n-shot examples from the dataset
    questions, answers, n_shot_examples = generate_n_shot_examples(
        questions, answers, num_examples
    )
    
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


def compute_perplexities_n_selfcertainties(
        model: torch.nn.Module,
        input_ids: torch.Tensor
    ) -> list:
    """
    Compute perplexities and self-certainties for a batch of input_ids using the provided model.
    Returns a list of perplexities and self-certainties for each sample in the batch.

    Args:
        model (torch.nn.Module): The language model to evaluate.
        input_ids (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing
                                  the input token IDs.
    Returns:
        tuple: A tuple containing two lists:
            - perplexities: A list of perplexity values for each sample in the batch.
            - self_certainty: A list of self-certainty values for each sample in the batch.
    """
    with torch.no_grad():
        model.eval()
        outputs = model(input_ids=input_ids)
    
    logits = outputs.logits
    labels = input_ids

    shift_logits = logits[..., :-1, :].contiguous()  # (bsz, seq_length-1, vocab_size)
    shift_labels = labels[..., 1:].contiguous()      # (bsz, seq_length-1)

    # Compute per sample in a batch
    vocab_size = shift_logits.size(-1)
    shift_logits = shift_logits.view(input_ids.size(0), -1, vocab_size)  # (bsz, seq_length-1, vocab_size)
    shift_labels = shift_labels.view(input_ids.size(0), -1)  # (bsz, seq_length-1)

    log_probs = F.log_softmax(shift_logits, dim=-1) # (bsz, seq_length-1, vocab_size)
    true_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)  # (bsz, seq_length-1)
    perplexities = torch.exp(-true_log_probs.mean(dim=-1))  # (bsz,)

    uniform_probs = F.softmax(torch.ones_like(shift_logits) / vocab_size, dim=-1)  # Uniform distribution over vocab
    kl = F.kl_div(log_probs, uniform_probs, reduction='none')  # (bsz, seq_length-1, vocab_size)
    
    # Token-level kl is the sum across vocab size for each token
    kl = kl.sum(dim=-1)  # (bsz, seq_length-1)

    # Self-certainty is the average KL over tokens in a sequence
    self_certainty = kl.mean(dim=-1)  # (bsz,)

    assert perplexities.shape[0] == input_ids.shape[0], "Perplexities shape does not match input_ids batch size"
    return perplexities.tolist(), self_certainty.tolist()
