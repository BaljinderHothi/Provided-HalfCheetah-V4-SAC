import torch
import gymnasium as gym
import numpy as np
from actor_impl import Actor

""" 
This section sets up the basic configuration for evaluating the  AI agent.
It specifies the environment (HalfCheetah-v4), the file path for the pre-trained
model weights, how many test episodes to run, and which computing device to use
(GPU if available, otherwise CPU).
"""
ENV_NAME = "HalfCheetah-v4"
MODEL_WEIGHTS_PATH = "halfcheetah_v4_actor_weights.pt" 
NUM_EVALUATION_EPISODES = 10
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_agent(env_name, actor_weights_path, num_episodes, seed, device):
    """
    This function evaluates how well the AI agent performs in the specified
    environment. It runs the agent through multiple episodes and records the total
    rewards achieved. The function handles loading the agent's neural network
    weights, simulating the environment, and collecting performance statistics.
    """
    print(f"Using device: {device}")

    eval_env = gym.vector.make(env_name, num_envs=1, asynchronous=False)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    actor = Actor(eval_env).to(device)
    
    try:
        actor.load_state_dict(torch.load(actor_weights_path, map_location=device))
        print(f"Successfully loaded weights from {actor_weights_path}")
    except FileNotFoundError:
        print(f"Error: Weights file not found at {actor_weights_path}")
        print("Please ensure the MODEL_WEIGHTS_PATH is correct and the file exists.")
        print("Running with uninitialized weights, which will not give meaningful results.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Running with uninitialized weights.")

    actor.eval()

    total_rewards = []
    print(f"\nEvaluating for {num_episodes} episodes...")

    """
    This loop runs multiple test episodes, where for each episode:
    1. The environment is reset to a starting state
    2. The agent observes the environment and chooses actions
    3. Actions are applied and rewards are collected
    4. The process repeats until the episode ends
    5. The total reward for the episode is recorded
    """
    for episode in range(num_episodes):
        obs, info = eval_env.reset(seed=seed + episode) 
        
        done_flag_for_vector_env = np.array([False]) 
        episode_reward = 0.0
        
        while not done_flag_for_vector_env[0]:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                _, _, deterministic_action = actor.get_action(obs_tensor)
            
            action_np = deterministic_action.cpu().numpy() 
            
            next_obs, reward, terminated, truncated, info = eval_env.step(action_np)
            
            done_flag_for_vector_env[0] = terminated[0] or truncated[0]
            
            episode_reward += reward[0]
            obs = next_obs
            
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes} | Reward: {episode_reward:.2f}")

    eval_env.close()
    
    """
    This section calculates summary statistics about the agent's performance
    across all test episodes, including average reward, standard deviation,
    and the minimum and maximum rewards achieved.
    """
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    min_reward = np.min(total_rewards)
    max_reward = np.max(total_rewards)

    print("\n--- Evaluation Summary ---")
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"Min Reward: {min_reward:.2f}")
    print(f"Max Reward: {max_reward:.2f}")

    """
    This section reports the evaluation results. The goal is to maximize
    the average reward - higher values indicate better performance of the model.
    No specific target is set as we want the model to achieve the best possible performance.
    """
    print(f"Average reward: {avg_reward:.2f}")
    print(f"------------------------") # just so i can read the terminal better
    
    return avg_reward, std_reward
if __name__ == "__main__":
    evaluate_agent(ENV_NAME, MODEL_WEIGHTS_PATH, NUM_EVALUATION_EPISODES, SEED, DEVICE)