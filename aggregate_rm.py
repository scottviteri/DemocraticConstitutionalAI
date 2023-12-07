from train_rm import load_model_with_reward_head

def generate_aggregate_reward_model(scores):
    def aggregate_reward_model(input):
        aggregate_reward = 0
        for constitution_id, score in scores.items():
            model = load_model_with_reward_head(f'rm_weights/{constitution_id}_rm_weights.pth')
            outputs = model(input, output_hidden_states=True)
            reward = model.reward_head(outputs.hidden_states[-1][0,-1,:])
            aggregate_reward += score * reward
        return aggregate_reward
    return aggregate_reward_model

# offline: 
# 1. train a reward model for each constitution
# 2. save the weights of each reward model
# online:
# 1. receive a user proposed constitution
# 2. calculate a similarity score between each constitution and the user proposed constitution
# 3. using the saved reward model weights, compute an aggregate reward weighted by similarity score