from train_rm import load_model_with_reward_head
from utils import rm_weight_path

def generate_aggregate_reward_model(scores):
    def aggregate_reward_model(input):
        aggregate_reward = 0
        for constitution_id, score in scores.items():
            model = load_model_with_reward_head(rm_weight_path(constitution_id))
            outputs = model(input, output_hidden_states=True)
            reward = model.reward_head(outputs.hidden_states[-1][0,-1,:])
            aggregate_reward += score * reward
        return aggregate_reward
    return aggregate_reward_model
