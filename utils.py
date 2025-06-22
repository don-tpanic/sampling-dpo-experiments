import torch
import torch.nn.functional as F


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


def compute_perplexities_n_selfcertainties(
        model: torch.nn.Module,
        input_ids: torch.Tensor
    ) -> list:
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
