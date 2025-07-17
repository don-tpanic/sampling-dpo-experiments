This repository contains experiments applying Direct Preference Optimization (DPO) and its variants to improve the mathematical reasoning abilities of SFT-trained language models. The approach includes techniques like rejection sampling and self-certainty for selecting preference pairs. The repository is loosely organized to allow fast iteration and experimentation, driven primarily by personal interest in related topics. While contributions and pull requests are welcome, the project is not currently intended to be a fully collaborative effort. Feel free to reach out if you'd like to discuss ideas or directions.

## Overview
We investigate using self-certainty (a model's internal confidence measure; inspired by [this work](https://arxiv.org/pdf/2502.18581) and [this work](https://arxiv.org/abs/2505.19590)) to select preference pairs for DPO training on mathematical reasoning tasks, such as GSM8K.

### Training Method
- LoRA + DPO + Additional regularization losses to mitigate model collapse e.g., [NLL](https://arxiv.org/abs/2403.07691) (cf. ORPO) and [DPOP](https://arxiv.org/abs/2402.13228).

### Results
- **Baseline**: Phi-3.5-mini-instruct accuracy on GSM8K: 73.2%
- **DPO (LoRA, 1 epoch)**: 79.1% accuracy
- **Statistical significance**: p-value < 0.001 (McNemar's test)

### Findings
1. **Self-certainty as quality signal**: Confirmed that correct answers have significantly higher self-certainty than incorrect ones
2. **DPO degradation**: Vanilla DPO performance drops to 2.65% after 2 epochs, confirming known instability issues
3. **Mitigation strategies**: Preliminary experiments with DPOP loss and NLL regularization show promise for reducing degradation

### Future Work
- Higher quality solution sampling (prevent correct-answer-incorrect-reasoning problems due to LLM backtracking, e.g., [Table 7](https://arxiv.org/abs/2308.01825))
- Improved preference pair selection using multiple factors
- Process-level and token-level reward signals
- Testing on more challenging mathematical datasets

## Implementation

### Repo structure 
* `explore.ipynb`: initial exploratory notebook understanding data (e.g., GSM8K and how baseline model such as phi-3.5 performs).
* `prepare_training_data.py`: samples positive and negative examples for DPO training, where each sample has associated data fields such as logprobs.
* `train.py`: training script for DPO.
* `utils.py`: functions range from producing logprobs of full/answer-only sequences to config loading and sampling n-shot examples.
* `configs/`: various configurations.
* `data/`: filtered training data.
* `outputs/`: model eval outputs for post-training analysis.
* `eval.py`: evaluation script producing outputs.
* `eval.ipynb`: evaluation notebook for post-training analysis.
* `playground/`: testing toy examples and debug notes.

### Produce experiment outputs for training and analysis
* Prepare training data: `python prepare_training_data.py`
* Train using DPO: `python train.py --config <config_id>`
* Eval trained model: `python eval.py --peft_model_path <lora_adapter_path>`

### Results (figs and analyses)
* `figs/`
* `eval.ipynb`

