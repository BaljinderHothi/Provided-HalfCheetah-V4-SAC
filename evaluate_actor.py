import argparse
import gymnasium as gym
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter

#importing the actor
from actor_impl import Actor


"""
this function aims to 
Create a gym environment with standard wrappers for consistent evaluation:
- flattens observations for consistent input shape
- records episode statistics for monitoring
- clips actions to valid ranges
- normalizes observations and rewards to ensure consistent scaling
- clips extreme values to prevent numerical issues
"""
def make_env(env_id, seed, capture_video=False, run_name=None):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            if run_name:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


"""
Here i am trying to evaluate the actor
- loads weights from a checkpoint file
- runs multiple evaluation episodes and reports statistics
- optionally records videos of the policy in action
- returns detailed performance metrics
"""
def evaluate_actor(
    weights_path,
    env_id="HalfCheetah-v4",
    seed=42,
    num_episodes=10,
    capture_video=False,
    cuda=True,
    run_name=None,
    verbose=True,
):
    
   
    # preparing the environment, device, and model for evaluation.
    
    if run_name is None:
        run_name = f"{env_id}_eval_{int(time.time())}"
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    if verbose:
        print(f"Using device: {device}")
        print(f"Loading weights from: {weights_path}")
    
    # Create environment
    env = make_env(env_id, seed, capture_video, run_name)()
    
    # Initialize actor and load weights
    actor = Actor(env).to(device)
    actor.load_state_dict(torch.load(weights_path, map_location=device))
    actor.eval()
    
    # Initialize tensorboard writer for logging
    writer = SummaryWriter(f"eval_runs/{run_name}")
    
    """
    Evaluation Loop
    --------------
    Runs the loaded policy through multiple episodes and collects performance metrics.
    No training updates are performed - this is purely for evaluation.
    """
    episode_rewards = []
    episode_lengths = []
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Get action from the trained policy
            with torch.no_grad():
                action, _, _ = actor.get_action(torch.FloatTensor(obs).unsqueeze(0).to(device))
            
            # Execute action in the environment
            obs, reward, terminated, truncated, _ = env.step(action[0].cpu().numpy())
            
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if verbose:
            print(f"Episode {episode}/{num_episodes}, Reward: {total_reward:.2f}, Length: {steps}")
        
        # Log to tensorboard
        writer.add_scalar("evaluation/episode_reward", total_reward, episode)
        writer.add_scalar("evaluation/episode_length", steps, episode)
    
    """
    Results Compilation
    -----------------
    Calculate and report statistics on the policy's performance across episodes.
    """
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    elapsed_time = time.time() - start_time
    
    # Log summary statistics
    writer.add_scalar("evaluation/mean_reward", mean_reward, 0)
    writer.add_scalar("evaluation/std_reward", std_reward, 0)
    writer.add_scalar("evaluation/min_reward", min_reward, 0)
    writer.add_scalar("evaluation/max_reward", max_reward, 0)
    writer.add_scalar("evaluation/mean_length", mean_length, 0)
    
    # Print summary
    if verbose:
        print("\n" + "="*50)
        print(f"Evaluation Summary for {env_id}")
        print("="*50)
        print(f"Total episodes: {num_episodes}")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Min/Max reward: {min_reward:.2f}/{max_reward:.2f}")
        print(f"Mean episode length: {mean_length:.2f}")
        print(f"Evaluation time: {elapsed_time:.2f} seconds")
        print("="*50)
    
    # Close environment and writer
    env.close()
    writer.close()
    
    # Return evaluation metrics
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "mean_length": mean_length,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


"""
Command Line Interface
--------------------
Allows running the evaluation script from the command line with various configuration options.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained actor policy")
    parser.add_argument("--weights", type=str, required=True, help="Path to the pre-trained weights file")
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4", help="Gym environment ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--capture-video", action="store_true", help="Capture video of the episodes")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this evaluation run")
    
    args = parser.parse_args()
    
    evaluate_actor(
        weights_path=args.weights,
        env_id=args.env_id,
        seed=args.seed,
        num_episodes=args.episodes,
        capture_video=args.capture_video,
        cuda=not args.no_cuda,
        run_name=args.run_name,
    )