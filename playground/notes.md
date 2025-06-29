### Regarding the RFT paper:
* Unfair comparison: SFT has fewer training data than RFT
* I have doubt on how many "augmented" samples are mere format differences/word choices rather than true reasoning.
* Table 7: example2 is in fact shows interesting case where the paths are incorrect but final answer is correct - model is likely remembering the answer and trying to work backward to get the answer.

### Clarification questions
* Should I start off with reported 8-shot cot gsm8k results?
* The current problem setting doesn't support RFT because I am asked for one solution and one incorrect
    instead of many distinct answers.
* It sounds like a natural thing to try is DPO to enhance model's ability of disconerning good/bad examples.

### Planned steps:
* Figure out what is a different solution?
* Figure out how to choose best solutions? (self-certainty?)
* Adjusting temperature to see how much different?
* Implement DPO
* baseline1: phi3.5
* baseline2: lora fine-tuned on gsm8k?
* compare1: lora finetuned on gsm8k + augmented samples?
* compare2: lora finetuned on DPO (correct + augmented?)

### Thinking/concerns:
* phi-3.5 already trained on gsm8k most likely.
* Concern that incorrect is already unlikely, and force mistake (increase temp will cause format contrastives not reasoning contrastives)
* Given phi-3.5 is good already, we likely get fewer incorrect answers, and the distinct correct answers are likely more format than true reasoning difference.
* [data-problem] incorrect due to cutoff -> Really need step-level correctness 
* [data-problem] incorrect but mistakes are the same, not very diverse (i guess the problem is easy; sometimes often is because model ignores conditions.)

### Low-level code optimization
* (Done) parallelise question * sample generation
* (Done) Generation + ppl & sc eval in one forward pass

### Encountered lower-priority problems for later:
* Reliability of format following, more robust parsing
* We need to measure during eval, how many times the answer gets cutoff (this is more of a problem when we set max_new_tokens)



0shot: 72%
2shot: 75%
