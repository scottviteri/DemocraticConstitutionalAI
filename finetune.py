
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel, GPT2Model
from train_rm import load_model_with_reward_head
from tqdm import tqdm
import csv
from sklearn.model_selection import train_test_split
from utils import generated_preferences_path, rm_weight_path
import torch

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = GPT2LMHeadModel.from_pretrained(model_name)

rm_path = rm_weight_path(0)
reward_model = load_model_with_reward_head(rm_path)

context, resp1, resp2, gpt4_pref = "Context", "Response_1", "Response_2", "GPT4_Preference"

constitution_id = 0
csv_dicts = [] 
with open(generated_preferences_path(constitution_id), 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    print(next(reader))
    for row in reader:
        csv_dicts.append({context: row[1], resp1: row[2], resp2: row[3], gpt4_pref: row[4]})

train_data, val_data = train_test_split(csv_dicts, test_size=0.2, random_state=42)
optimizer = torch.optim.Adam(model.parameters())

for epoch in tqdm(range(1)):  # number of epochs can be adjusted
    for i, data in enumerate(tqdm(train_data[:len(train_data)])):
        context = data["Context"]
        context_encoded = tokenizer.encode(context, return_tensors='pt')
        continuation = model.generate(context_encoded, do_sample=True, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
        input_continuation = torch.cat([context_encoded, continuation], dim=-1)
        # Check if input_continuation is longer than the context window
        if input_continuation.size(1) > model.config.n_ctx: continue
        # Forward pass through the model and log hidden states
        outputs_continuation = model(input_continuation, output_hidden_states=True)
        # Get the reward prediction
        reward = reward_model.reward_head(outputs_continuation.hidden_states[-1][0,-1,:]).item()
        # If reward is greater than a threshold, finetune the model on the continuation given the context. Otherwise, continue
        if reward > 0.5:
            model.train()
            loss = model(input_continuation, labels=input_continuation)[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            continue

torch.save(model.state_dict(), f'finetuned_weights/{constitution_id}_weights.pth')
