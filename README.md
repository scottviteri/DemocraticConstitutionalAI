Overall goal is to
A) Get a weight for each constitution 
B) Train a reward model for each constitution
C) Train gpt2 with the weighted reward models

Pragmatic steps to get there by mid December:
1. Train gpt2 on a reward model from huggingface (done)
2. Use a constitution + gpt4 calls to train a reward model (done)
3. Do this for each constitution
4. Train gpt2 according the mixed model

Run experiments


Pragmatic step 2 in more detail
1. Load the HH RLHF dataset from huggingface 
2. Use gpt2 to generate pairs of completions
3. Get gpt4 to pick which generation it likes better
