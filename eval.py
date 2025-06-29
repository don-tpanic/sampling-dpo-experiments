import argparse
import os
import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from . import utils


def load_peft_model(
        base_model_path: str,
        peft_model_path: str,
    ):
    """
    Load a PEFT (LoRA) model for testing.
    """
    from peft import PeftModel

    # Load the base model
    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    
    # Load the PEFT model (LoRA weights)
    print(f"Loading PEFT adapter from {peft_model_path}")
    model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = model.merge_and_unload()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="Path to the base model (e.g., 'microsoft/Phi-3.5-mini-instruct').",
    )
    parser.add_argument(
        "--peft_model_path",
        type=str,
        default=None,
        help="Path to the PEFT model (LoRA weights). If None, will use the base model directly.",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=None,
        help="GPU ID to use for evaluation (e.g., '0' for the first GPU). If None, defaults to '0'.",
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id if args.gpu_id else "0"

    base_model_path = args.base_model_path
    dataset = datasets.load_dataset("openai/gsm8k", "main")

    # Generate n-shot examples from the dataset
    _, _, n_shot_examples = utils.generate_n_shot_examples(
        dataset['train']['question'], dataset['train']['answer'], 2
    )

    prompt_template = """You are given a math question. You must provide a concise step-by-step reasoning
        and a final answer. Your response should follow strictly the format of the provided examples where each new line is a reasoning step
        written in a very concise style, and the final answer is on the last line. There should be roughly 2-4 steps, but it is okay
        to have more or less steps if needed.
        
        {n_shot_examples}

        # Question:
        {question}
    """
    if not args.peft_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    else:
        model, tokenizer = load_peft_model(
            base_model_path=base_model_path,
            peft_model_path=args.peft_model_path,
        )

    acc, outputs = utils.evaluate_model(
        model=model,
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        max_new_tokens=256,
        temperature=0.7,
        batch_size=8,
        n_shot_examples=n_shot_examples,
    )
    
    # Save outputs baswed on model
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    if not args.peft_model_path:
        output_file = os.path.join(output_dir, f"gsm8k_eval_{base_model_path.split('/')[-1]}.pt")
    else:
        # Concat base+peft model
        base = base_model_path.split('/')[-1]
        peft_config = args.peft_model_path.split('/')[1]
        peft_run = args.peft_model_path.split('/')[2]
        peft_ckpt = args.peft_model_path.split('/')[-1]
        output_file = os.path.join(output_dir, f"gsm8k_eval_{base}_{peft_config}_{peft_run}_{peft_ckpt}.pt")
    torch.save(outputs, output_file)
    print(f"Evaluation results saved to {output_file}")
