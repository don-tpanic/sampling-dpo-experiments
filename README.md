# sampling-dpo-experiments

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

