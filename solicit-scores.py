#import openai
#
#import os
#openai.api_key = os.getenv('OPENAI_API_KEY')
#
#response = openai.ChatCompletion.create(
#  model="gpt-4",
#  messages=[
#        {"role": "system", "content": "You are a helpful assistant."},
#        {"role": "user", "content": "Translate this English text to French: 'Hello, how are you?'"}
#    ]
#)
#
## constitution as a bag of values
## sometimes constitutions is an implementation of values (in more detail) -- eg a reconciliation
#group_constitution = {
#    "Youth and Digital Natives": "AI should prioritize educational content and learning opportunities, ensuring accessibility for all.. It must respect and protect the privacy of young users, with robust data security measures. It should also promote mental health awareness and provide resources for young people dealing with stress and anxiety.",
#    "Elderly Population": "AI should have an intuitive and easy-to-use interface for the elderly, accommodating for potential physical and cognitive limitations. It should assist in monitoring health, reminding of medications, and providing easy access to medical information and support. It should also help in maintaining social connections, offering platforms for communication with family and friends.",
#    "Cultural and Ethnic Minorities": "AI must be designed to be culturally sensitive, respecting and reflecting diverse cultural backgrounds and languages. It should actively work against biases, ensuring fair and equal treatment for all ethnic and cultural groups. It should also include diverse voices in its development and offer content that represents a wide range of cultures and ethnicities.",
#    "Women and Gender Minorities": "AI should promote gender equity, ensuring equal opportunities and treatment for all genders. It must prioritize the safety and security of women and gender minorities, including features that protect against harassment and abuse. It should also empower women and gender minorities, providing platforms for their voices and stories.",
#    "People with Disabilities": "AI must be fully accessible, with features that accommodate various types of disabilities. It should serve as an assistive tool, aiding in daily tasks and enhancing the independence of individuals with disabilities. It should also be developed with input from people with disabilities, ensuring that it meets their unique needs and preferences."
#}
#
#print(group_constitution)
#
#from openai import OpenAI
#client = OpenAI()
#
#response = client.embeddings.create(
#    input="Your text string goes here",
#    model="text-embedding-ada-002"
#)
#
#print(response.data[0].embedding)
#
#

from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    #api_key="My API Key",
)

response = client.models.list()

#for model in response['data']:
#    print(model['id'])

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4-1106-preview",
)

print(chat_completion)
