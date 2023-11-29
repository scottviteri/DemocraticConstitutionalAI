from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch import nn
import csv

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# next want to train rm -- could do using an extra head on gpt2
class RewardHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(model.config.n_embd, 1)

    def forward(self, features):
        return self.linear(features)

reward_head = RewardHead()
model.add_module("reward_head", reward_head)

def load_model_with_reward_head(path: str):
    model_copy = GPT2LMHeadModel.from_pretrained("gpt2")
    reward_head_copy = RewardHead()
    model_copy.add_module("reward_head", reward_head_copy)
    model_copy.load_state_dict(torch.load(path))
    return model_copy



csv_dicts = [] 
with open('updated_responses.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    print(next(reader))
    for row in reader:
        csv_dicts.append({'Context': row[1], 'Response 1': row[2], 'Response 2': row[3], 'gpt4 preference': row[4]})

print(csv_dicts[:3])

optimizer = torch.optim.Adam(model.parameters())
def criterion(preferred_reward, not_preferred_reward):
    return - torch.log(torch.sigmoid(preferred_reward - not_preferred_reward))

from tqdm import tqdm

from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(csv_dicts, test_size=0.2, random_state=42)

for epoch in tqdm(range(1)):  # number of epochs can be adjusted
    for i, data in enumerate(tqdm(train_data[:len(train_data)])):
        context = data['Context']
        response_1 = data['Response 1']
        response_2 = data['Response 2']
        preference = int(data['gpt4 preference'])

        # Tokenize and encode the context and responses
        context_encoded = tokenizer.encode(context, return_tensors='pt')
        response_1_encoded = tokenizer.encode(response_1, return_tensors='pt')
        response_2_encoded = tokenizer.encode(response_2, return_tensors='pt')

        # Concatenate the context with the responses
        input_1 = torch.cat([context_encoded, response_1_encoded], dim=-1)
        input_2 = torch.cat([context_encoded, response_2_encoded], dim=-1)

        # Check if input_1 or input_2 is longer than the context window
        if input_1.size(1) > model.config.n_ctx or input_2.size(1) > model.config.n_ctx:
            continue

        # Forward pass through the model and log hidden states
        outputs_1 = model(input_1, output_hidden_states=True)
        outputs_2 = model(input_2, output_hidden_states=True)

        # Get the reward predictions
        reward_1 = model.reward_head(outputs_1.hidden_states[-1][0,-1,:])
        reward_2 = model.reward_head(outputs_2.hidden_states[-1][0,-1,:])

        # Calculate the loss
        if preference == 1:
            loss = criterion(reward_1, reward_2)
        else:
            loss = criterion(reward_2, reward_1)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print validation loss every 5 datapoints
        if i % 5 == 0:
            val_loss = 0
            with torch.no_grad():
                for data in val_data:
                    context = data['Context']
                    response_1 = data['Response 1']
                    response_2 = data['Response 2']
                    preference = int(data['gpt4 preference'])

                    # Tokenize and encode the context and responses
                    context_encoded = tokenizer.encode(context, return_tensors='pt')
                    response_1_encoded = tokenizer.encode(response_1, return_tensors='pt')
                    response_2_encoded = tokenizer.encode(response_2, return_tensors='pt')

                    # Concatenate the context with the responses
                    input_1 = torch.cat([context_encoded, response_1_encoded], dim=-1)
                    input_2 = torch.cat([context_encoded, response_2_encoded], dim=-1)

                    # Check if input_1 or input_2 is longer than the context window
                    if input_1.size(1) > model.config.n_ctx or input_2.size(1) > model.config.n_ctx:
                        continue

                    # Forward pass through the model and log hidden states
                    outputs_1 = model(input_1, output_hidden_states=True)
                    outputs_2 = model(input_2, output_hidden_states=True)

                    # Get the reward predictions
                    reward_1 = model.reward_head(outputs_1.hidden_states[-1][0,-1,:])
                    reward_2 = model.reward_head(outputs_2.hidden_states[-1][0,-1,:])

                    # Calculate the loss
                    if preference == 1:
                        loss = criterion(reward_1, reward_2)
                    else:
                        loss = criterion(reward_2, reward_1)

                    val_loss += loss.item()

            print(f"Epoch {epoch+1}, Datapoint {i+1}, Validation Loss: {val_loss/len(val_data)}")


        # Save the model weights
        torch.save(model.state_dict(), 'model_weights.pth')
