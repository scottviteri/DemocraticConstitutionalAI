from utils import constitutions
from generate_preferences import generate_preferences
from train_rm import train_rm
from solicit_scores import get_normalized_scores
from aggregate_rm import generate_aggregate_reward_model

# preference generation
for i in range(len(constitutions)):
    generate_preferences(i)

# reward model training
for i in range(len(constitutions)):
    train_rm(i)

proposed_constitution = "AI should be designed to be inclusive, respecting and reflecting diverse backgrounds and languages. It should actively work against biases, ensuring fair and equal treatment for all groups. It should also include diverse voices in its development and offer content that represents a wide range of cultures, ethnicities, and abilities."

scores = get_normalized_scores(proposed_constitution, constitutions)

aggregate_rm = generate_aggregate_reward_model(scores)

