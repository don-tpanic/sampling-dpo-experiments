import torch
import datasets
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import utils

"""
Prepare data for DPO training.
"""

def generate_trainset_ids_n_metrics(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset: datasets.Dataset,
        num_questions: int = 1,
        batch_size: int = 8,
    ) -> dict:
    """
    Run through the dataset and convert the ground truth answers to gen_ids,
    and compare the logprobs and self-certainties of the ground truth answers.

    This function is needed because, we require gen_ids and metrics when during
    synthetic data generation, there might be questions the model never answered
    correctly.
    """
    questions = dataset['question']
    answers = dataset['answer']

    # Randomly sample `num_questions` questions from the dataset
    questions, answers, n_shot_examples = utils.generate_n_shot_examples(
        questions, answers, num_examples=num_questions
    )

    # Prepare all (question_idx, sample_idx) pairs
    questions = questions[:num_questions]
    answers = answers[:num_questions]

    all_logprobs = {}
    all_selfcertainties = {}
    all_gen_ids = {}

    for batch_start in tqdm.tqdm(range(0, num_questions, batch_size), desc="Generating trainset ids and metrics"):
        batch_end = min(batch_start + batch_size, num_questions)
        batch_questions = questions[batch_start:batch_end]
        batch_answers = answers[batch_start:batch_end]

        # Prepare batch inputs - PROMPT ONLY (without answers)
        batch_chats_prompt_only = []
        for question in batch_questions:
            chat = [
                {
                    "role": "user",
                    "content": prompt_template.format(question=question, n_shot_examples=n_shot_examples)
                }
            ]
            batch_chats_prompt_only.append(chat)

        # Prepare batch inputs - FULL (with answers)
        batch_chats_full = []
        for question, answer in zip(batch_questions, batch_answers):
            chat = [
                {
                    "role": "user",
                    "content": prompt_template.format(question=question, n_shot_examples=n_shot_examples)
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
            batch_chats_full.append(chat)

        # Format chats - PROMPT ONLY
        formatted_chats_prompt_only = [
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            for chat in batch_chats_prompt_only
        ]
        
        # Format chats - FULL
        formatted_chats_full = [
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            for chat in batch_chats_full
        ]

        # Tokenize both versions WITHOUT padding first to get individual lengths
        prompt_only_tokens = [
            tokenizer(chat, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
            for chat in formatted_chats_prompt_only
        ]
        
        full_tokens = [
            tokenizer(chat, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
            for chat in formatted_chats_full
        ]

        # Calculate answer start positions and lengths for each sequence
        answer_start_positions = [len(prompt_tokens) for prompt_tokens in prompt_only_tokens]
        answer_lengths = [len(full_tokens[i]) - answer_start_positions[i] for i in range(len(full_tokens))]

        # Tokenize batch
        batch_inputs = tokenizer(
            formatted_chats_full,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to("cuda")
        input_ids = batch_inputs.input_ids
        attention_mask = batch_inputs.attention_mask

        # Forward pass through the whole batch
        # in order to compute logprobs and self-certainties
        # (ignore prompt and pad tokens in the answer)
        # Compute logprobs and self-certainties
        logprobs, selfcertainties = utils.forward_pass_compute_logprobs_n_selfcertainties(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            answer_start_positions=answer_start_positions,
            answer_lengths=answer_lengths,
        )

        # Store results in dictionaries
        for i in range(batch_size):
            q_idx = batch_start + i
            if q_idx not in all_logprobs:
                all_logprobs[q_idx] = []
                all_selfcertainties[q_idx] = []
                all_gen_ids[q_idx] = []

            # Get the generated tokens (answer part)
            gen_ids = input_ids[i, answer_start_positions[i]:]
            all_gen_ids[q_idx].append(gen_ids)

            # Store logprobs and self-certainties
            all_logprobs[q_idx].append(logprobs[i])
            all_selfcertainties[q_idx].append(selfcertainties[i])

    results = {
        "logprobs": all_logprobs,
        "selfcertainties": all_selfcertainties,
        "gen_ids": all_gen_ids,
    }
    torch.save(results, "trainset_ids_n_metrics.pt")


def generate_synthetic_data(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset: datasets.Dataset,
        num_samples_per_question: int = 2,
        num_questions: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        num_examples: int = 8,
        batch_size: int = 8,
    ) -> dict:
    """
    Generate synthetic data with full batching across questions and samples.
    """
    questions = dataset['question']
    answers = dataset['answer']

    # Randomly sample `num_examples` examples from the dataset
    # both questions and answers to create n-shot examples
    questions, answers, n_shot_examples = utils.generate_n_shot_examples(
        questions, answers, num_examples=num_examples
    )

    # Prepare all (question_idx, sample_idx) pairs
    questions = questions[:num_questions]
    answers = answers[:num_questions]
    all_pairs = []
    for q_idx in range(num_questions):
        for s_idx in range(num_samples_per_question):
            all_pairs.append((q_idx, s_idx))
    
    total_pairs = len(all_pairs)

    # Initialize result containers
    all_solutions = {}
    all_correctness = {}
    all_logprobs = {}
    all_selfcertainties = {}
    all_input_ids = {}
    all_gen_ids = {}

    for batch_start in tqdm.tqdm(range(0, total_pairs, batch_size), desc="Generating synthetic data"):
        batch_end = min(batch_start + batch_size, total_pairs)
        batch_pairs = all_pairs[batch_start:batch_end]

        # Prepare batch inputs
        batch_chats = []
        batch_question_indices = []
        for q_idx, s_idx in batch_pairs:
            question = questions[q_idx]
            chat = [
                {
                    "role": "user",
                    "content": prompt_template.format(question=question, n_shot_examples=n_shot_examples)
                }
            ]
            batch_chats.append(chat)
            batch_question_indices.append(q_idx)

        # Format all chats
        formatted_chats = [
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            for chat in batch_chats
        ]

        # Tokenize batch
        batch_inputs = tokenizer(
            formatted_chats,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to("cuda")
        input_ids = batch_inputs.input_ids
        attention_mask = batch_inputs.attention_mask

        # Generate answers for the batch, also return
        # logits for perplexity and self-certainty computation
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,  # handled explicitly
                pad_token_id=tokenizer.pad_token_id,
                output_logits=True,
                return_dict_in_generate=True,
            )

        output_ids = outputs.sequences  # shape: (bsz, seq_len)
        logits = outputs.logits         # tuple of (bsz, vocab_size), length = num_new_tokens

        # Convert logits to tensor (bsz, num_new_tokens, vocab_size)
        # in order to compute logprobs and self-certainties
        logits = torch.stack(logits, dim=1)  # shape: (bsz, num_new_tokens, vocab_size)
        gen_ids = output_ids[:, input_ids.shape[1]:]  # only generated tokens, shape: (bsz, num_new_tokens)
        assert gen_ids.shape[1] == logits.shape[1], \
            "Generated tokens and logits must match in length."

        # Compute logprobs and self-certainties
        logprobs, selfcertainties = utils.compute_logprobs_n_selfcertainties(
            logits=logits,
            gen_ids=gen_ids,
            tokenizer=tokenizer,
        )

        # Process results and organize by question
        for i, (q_idx, s_idx) in enumerate(batch_pairs):
            input_length = input_ids.shape[1]
            generated_tokens = output_ids[i, input_length:]
            ans_pred = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            if q_idx not in all_solutions:
                all_solutions[q_idx] = []
                all_correctness[q_idx] = []
                all_logprobs[q_idx] = []
                all_selfcertainties[q_idx] = []
                all_input_ids[q_idx] = []
                all_gen_ids[q_idx] = []

            all_solutions[q_idx].append(ans_pred)
            all_correctness[q_idx].append(utils.is_correct_solution(ans_pred, answers[q_idx]))
            all_logprobs[q_idx].append(logprobs[i])
            all_selfcertainties[q_idx].append(selfcertainties[i])
            all_input_ids[q_idx].append(input_ids[i])
            all_gen_ids[q_idx].append(gen_ids[i])

        del input_ids, output_ids, attention_mask
        torch.cuda.empty_cache()
    
    results = {
        "solutions": all_solutions,
        "correctness_mask": all_correctness,
        "logprobs": all_logprobs,
        "selfcertainties": all_selfcertainties,
        "input_ids": all_input_ids,
        "gen_ids": all_gen_ids,
    }

    torch.save(results, "synthetic_data.pt")


def prepare_training_data(
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        synthetic_data: dict
    ):
    """
    Based on the generated synthetic data, prepare training data for model training.

    synthetic_data = {
        "solutions": all_solutions,
        "correctness_mask": all_correctness,
        "logprobs": all_logprobs,
        "selfcertainties": all_selfcertainties,
        "questions": questions,
        "answers": answers,
        "input_ids": all_input_ids,
        "gen_ids": all_gen_ids,
    }
    
    where
        solutions: dict of {question_idx: [solution1, solution2, ...]}
        correctness_mask: dict of {question_idx: [True/False, True/False, ...]}
        logprobs: dict of {question_idx: [logprob1, logprob2, ...]}
        selfcertainties: dict of {question_idx: [selfcertainty1, selfcertainty2, ...]}
        questions: list of questions
        answers: list of answers
        n_shot_examples: list of n-shot examples
        input_ids: dict of {question_idx: [input_ids1, input_ids2, ...]}
        gen_ids: dict of {question_idx: [gen_ids1, gen_ids2, ...]}
    
    For each question,
        If solutions all correct, chosen is the one with highest selfcertainty,
                                reject is the one with lowest selfcertainty.
        If solutions all incorrect, chosen is the original answer, 
                                reject is the one with highest selfcertainty.
        If solutions mixed, chosen is the correct one with highest selfcertainty,
                                reject is the incorrect one with highest selfcertainty.

    To compile the training data, for each question, we collect the following:
        - input_ids: the input_ids for the question
        - chosen_ids: the gen_ids of the chosen solution
        - reject_ids: the gen_ids of the rejected solution
        - chosen_logprobs: the logprobs of the chosen solution
        - reject_logprobs: the logprobs of the rejected solution
        - chosen_selfcertainties: the self-certainties of the chosen solution
        - reject_selfcertainties: the self-certainties of the rejected solution
    """
    prepared_data = {
        "input_ids": [],
        "chosen_ids": [],
        "reject_ids": [],
        "chosen_logprobs": [],
        "reject_logprobs": [],
        "chosen_selfcertainties": [],
        "reject_selfcertainties": [],
    }

    for q_idx in range(len(synthetic_data['questions'])):
        solutions = synthetic_data['solutions'][q_idx]
        correctness_mask = synthetic_data['correctness_mask'][q_idx]
        logprobs = synthetic_data['logprobs'][q_idx]
        selfcertainties = synthetic_data['selfcertainties'][q_idx]
        input_ids = synthetic_data['input_ids'][q_idx]
        gen_ids = synthetic_data['gen_ids'][q_idx]

        # Determine chosen and reject solutions based on correctness
        if all(correctness_mask):
            # All correct
            chosen_idx = selfcertainties.index(max(selfcertainties))
            reject_idx = selfcertainties.index(min(selfcertainties))
        elif not any(correctness_mask):
            # All incorrect
            chosen_idx = 0  # Original answer
            reject_idx = selfcertainties.index(max(selfcertainties))
            # TODO: here needs to read off from the original answer
        else:
            # Mixed correctness
            correct_indices = [i for i, correct in enumerate(correctness_mask) if correct]
            incorrect_indices = [i for i, correct in enumerate(correctness_mask) if not correct]

            if not correct_indices or not incorrect_indices:
                continue
            
            chosen_idx = max(correct_indices, key=lambda i: selfcertainties[i])
            reject_idx = max(incorrect_indices, key=lambda i: selfcertainties[i])
        
        # Collect data for the chosen and rejected solutions
        prepared_data["input_ids"].append(input_ids)
        prepared_data["chosen_ids"].append(gen_ids[chosen_idx])
        prepared_data["reject_ids"].append(gen_ids[reject_idx])
        prepared_data["chosen_logprobs"].append(logprobs[chosen_idx])
        prepared_data["reject_logprobs"].append(logprobs[reject_idx])
        prepared_data["chosen_selfcertainties"].append(selfcertainties[chosen_idx])
        prepared_data["reject_selfcertainties"].append(selfcertainties[reject_idx])


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

    ds = datasets.load_dataset("openai/gsm8k", "main")

    prompt_template = """You are given a math question. You must provide a concise step-by-step reasoning
        and a final answer. Your response should follow strictly the format of the provided examples where each new line is a reasoning step
        written in a very concise style, and the final answer is on the last line. There should be roughly 2-4 steps, but it is okay
        to have more or less steps if needed.
        
        {n_shot_examples}

        # Question:
        {question}
    """

    # Generate trainset ids and metrics
    trainset_ids_n_metrics = generate_trainset_ids_n_metrics(
        model=model,
        tokenizer=tokenizer,
        dataset=ds['train'],
        num_questions=8,
        batch_size=8,
    )

    # # Generate synthetic data
    # synthetic_data = generate_synthetic_data(
    #     model=model,
    #     tokenizer=tokenizer,
    #     dataset=ds['train'],
    #     num_samples_per_question=2,
    #     num_questions=10,
    #     temperature=0.7,
    #     max_new_tokens=256,
    #     num_examples=8,
    #     batch_size=8,
    # )