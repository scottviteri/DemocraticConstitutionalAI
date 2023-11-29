# In[2]:
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import openai
import os
import csv

# Load all helpfulness/harmlessness subsets
dataset_all = load_dataset("Anthropic/hh-rlhf")
# In[15]:
secondex = dataset_all['test'][1]
print(secondex['chosen'])
print(secondex['rejected'])

# In[10]:


def extract_context(example):
    lines = example['chosen'].strip().split('\n')

    # Find the last line where 'Human' speaks before the final 'Assistant' response
    last_human_line_index = None
    for i, line in enumerate(lines[::-1]):
        if line.startswith('Human:'):
            last_human_line_index = len(lines) - i - 1
            break

    # Extract context (up to the last 'Human' line)
    context = '\n'.join(lines[:last_human_line_index + 1])
    return context


# In[16]:


first_example_context = extract_context(dataset_all['test'][1])
print(first_example_context)


# In[76]:


model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_response(context, max_length=50, temperature=0.5):
    input_ids = tokenizer.encode(context, return_tensors='pt')

    output = model.generate(input_ids, 
                            max_length=max_length + len(input_ids[0]), 
                            temperature=temperature,
                            do_sample=True,  # this is for randomness, just temperature doesnt work
                            num_return_sequences=1,
                            no_repeat_ngram_size=2,
                            pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

context = first_example_context + "\n\nAssistant: "

response1 = generate_response(context, temperature=0.5)
response2 = generate_response(context, temperature=0.5)

print("Response 1:", response1)
print("Response 2:", response2)


# In[88]:
import csv
from tqdm import tqdm

with open('generated_responses.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Example ID', 'Context', 'Response 1', 'Response 2'])

    for i in tqdm(range(50), desc="Generating Responses"):
        example = dataset_all['train'][i]
        context = extract_context(example) + "\n\nAssistant: "

        response1 = generate_response(context, temperature=0.5)
        response2 = generate_response(context, temperature=0.5)

        writer.writerow([i, context, response1, response2])

print("Generation completed and saved to generated_responses.csv")


# In[96]:
#looking at the CSV: example id, context (last line of human), response 1, and response 2
# Open the CSV file for reading
with open('generated_responses.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)

    # Convert the reader to a list, skipping the header
    data = list(reader)[1:]  # Skip header row

    # Access the 50th row
    if len(data) >= 50:
        example_id, context, response1, response2 = data[30]  # Index 49 for the 50th row
        print(f"Example ID: {example_id}")
        print(f"Context: {context}")
        print(f"Response 1: {response1}")
        print(f"Response 2: {response2}")
    else:
        print("The file doesn't have 50 examples.")


# In[ ]:

from openai import OpenAI
client = OpenAI()

#for now, this is just an arbitrary method of prompting gpt 4 to decide between the two generations. 
#arbitrary method: which completion does gpt4 find funnier?
def evaluate_funnier(response1, response2, openai_api_key):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Just respond with a single number, either 1 or 2 corresponding to which of these two responses you think is funnier?\n\n1: {response1}\n\n2: {response2}"}
    ]

    client.api_key = openai_api_key

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # gpt 4 turob
            messages=messages,
            max_tokens=10
        )

        result = response.choices[0].message.content.strip()
        return 1 if "1" in result else 2 if "2" in result else 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OpenAI API key not set in environment variables.")

with open('generated_responses.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)
    next(reader)
    next(reader)
    next(reader)
    next(reader)
    first_row = next(reader)
    example_id, context, response1, response2 = first_row

funnier_response = evaluate_funnier(response1, response2, openai_api_key)
print(f"Example ID: {example_id}, Funnier Response: {funnier_response}")


# In[117]:


#new CSV with new column corresponding to which response GPT4 finds funnier (1 or 2)

updated_rows = []

with open('generated_responses.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)
    header.append("gpt4 preference") 
    updated_rows.append(header)

    for row in tqdm(reader, desc="Processing rows"):
        example_id, context, response1, response2 = row
        funnier_response = evaluate_funnier(response1, response2, openai_api_key)
        row.append(funnier_response)
        updated_rows.append(row)

with open('updated_responses.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(updated_rows)

print("CSV file updated with GPT-4 preferences.")


# %%
