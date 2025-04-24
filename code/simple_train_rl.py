import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
from collections import deque

from code.data_utils import PolypDataset
from code.enhanced_rl_agent import EnhancedPPOAgent
from code.segmentation_env import SegmentationEnvDataset
from code.enhanced_rl_env import EnhancedPolypSegmentationEnv, create_env_factory

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent for interactive segmentation')
    
    # Data parameters
    parser.add_argument('--image_dir', type=str, default='data/raw/Original',
                      help='Directory containing images')
    parser.add_argument('--mask_dir', type=str, default='data/raw/Ground Truth',
                      help='Directory containing ground truth masks')
    parser.add_argument('--results_dir', type=str, default='results/rl',
                      help='Directory to save results')
    
    # Environment parameters
    parser.add_argument('--max_steps', type=int, default=20,
                      help='Maximum number of steps per episode')
    parser.add_argument('--n_segments', type=int, default=100,
                      help='Number of superpixels for segmentation')
    
    # Agent parameters
    parser.add_argument('--hidden_size', type=int, default=128,
                      help='Hidden size of the agent')
    parser.add_argument('--lr', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=1000,
                      help='Number of episodes to train for')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for updates')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs per update')
    
    # Evaluation parameters
    parser.add_argument('--eval_interval', type=int, default=10,
                      help='Interval for evaluation')
    parser.add_argument('--save_interval', type=int, default=50,
                      help='Interval for saving the model')
    parser.add_argument('--num_eval_episodes', type=int, default=5,
                      help='Number of episodes for evaluation')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_dataset(args):
    """Load dataset from directory"""
    # Create dataset
    dataset = PolypDataset(
        data_dir=os.path.dirname(args.image_dir), 
        split='train'
    )
    
    val_dataset = PolypDataset(
        data_dir=os.path.dirname(args.image_dir), 
        split='val'
    )
    
    print(f"Loaded {len(dataset)} training samples and {len(val_dataset)} validation samples")
    
    return dataset, val_dataset

def create_environments(dataset, val_dataset, args):
    """Create environments for training and evaluation"""
    # Extract images and masks
    train_images = [sample['image'].numpy() for sample in dataset]
    train_masks = [sample['mask'].squeeze().numpy() for sample in dataset]
    
    val_images = [sample['image'].numpy() for sample in val_dataset]
    val_masks = [sample['mask'].squeeze().numpy() for sample in val_dataset]
    
    # Create environment factories
    train_env_factory = create_env_factory(
        train_images, 
        train_masks, 
        max_steps=args.max_steps,
        device=args.device
    )
    
    val_env_factory = create_env_factory(
        val_images, 
        val_masks, 
        max_steps=args.max_steps,
        device=args.device
    )
    
    return train_env_factory, val_env_factory

def create_agent(args, env):
    """Create agent for training"""
    # Get observation shape from the environment
    state = env.reset()[0]
    if isinstance(state, dict):
        # If state is a dictionary, concatenate all values
        state_tensors = []
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                state_tensors.append(torch.from_numpy(value).float())
        
        # Concatenate along channel dimension
        state_tensor = torch.cat(state_tensors, dim=0)
        obs_shape = state_tensor.shape
    else:
        # Otherwise, use state shape directly
        obs_shape = state.shape
    
    # Create agent
    agent = EnhancedPPOAgent(
        obs_shape=obs_shape,
        action_space=env.action_space.n,
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        device=torch.device(args.device)
    )
    
    return agent

def collect_rollout(env, agent):
    """Collect a single rollout (episode) from the environment"""
    state, info = env.reset()
    
    # Convert state to tensor if it's a dictionary
    if isinstance(state, dict):
        state_tensors = []
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                state_tensors.append(torch.from_numpy(value).float())
        
        # Concatenate along channel dimension
        state = torch.cat(state_tensors, dim=0)
    
    state = torch.tensor(state, dtype=torch.float32).to(agent.device)
    
    done = False
    episode_reward = 0
    episode_length = 0
    
    while not done:
        # Select action
        action = agent.select_action(state)
        
        # Take step in environment
        next_state, reward, done, info = env.step(action.item())
        
        # Convert next_state to tensor if it's a dictionary
        if isinstance(next_state, dict):
            next_state_tensors = []
            for key, value in next_state.items():
                if isinstance(value, np.ndarray):
                    next_state_tensors.append(torch.from_numpy(value).float())
            
            # Concatenate along channel dimension
            next_state = torch.cat(next_state_tensors, dim=0)
        
        next_state = torch.tensor(next_state, dtype=torch.float32).to(agent.device)
        
        # Store experience
        agent.store_experience(
            obs=state,
            action=action,
            reward=torch.tensor(reward, dtype=torch.float32).to(agent.device),
            next_obs=next_state,
            done=torch.tensor(done, dtype=torch.float32).to(agent.device)
        )
        
        # Update state and counters
        state = next_state
        episode_reward += reward
        episode_length += 1
    
    return episode_reward, episode_length, info

def evaluate_agent(agent, env_factory, num_episodes):
    """Evaluate agent on multiple episodes"""
    rewards = []
    dices = []
    ious = []
    
    for _ in range(num_episodes):
        env = env_factory()
        reward, _, info = collect_rollout(env, agent)
        rewards.append(reward)
        dices.append(info['dice'])
        ious.append(info['iou'])
    
    return {
        'reward': np.mean(rewards),
        'dice': np.mean(dices),
        'iou': np.mean(ious)
    }

def train(args):
    """Train the agent"""
    # Set random seed
    set_seed(args.seed)
    
    # Load dataset
    dataset, val_dataset = load_dataset(args)
    
    # Create environments
    train_env_factory, val_env_factory = create_environments(dataset, val_dataset, args)
    train_env = train_env_factory()
    
    # Create agent
    agent = create_agent(args, train_env)
    
    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Training variables
    best_reward = -float('inf')
    episode_rewards = []
    episode_dices = []
    
    # Training loop
    for episode in range(args.num_episodes):
        # Create a new environment for each episode
        env = train_env_factory()
        
        # Collect rollout
        reward, length, info = collect_rollout(env, agent)
        
        # Store metrics
        episode_rewards.append(reward)
        episode_dices.append(info['dice'])
        
        # Update agent
        if len(agent.experience_buffer) > args.batch_size:
            loss_info = agent.update(batch_size=args.batch_size, epochs=args.epochs)
            agent.experience_buffer = []  # Clear buffer after update
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_dice = np.mean(episode_dices[-10:])
            
            print(f"Episode {episode+1}/{args.num_episodes}: "
                  f"Reward = {avg_reward:.4f}, Dice = {avg_dice:.4f}")
        
        # Evaluation
        if (episode + 1) % args.eval_interval == 0:
            eval_metrics = evaluate_agent(
                agent=agent,
                env_factory=val_env_factory,
                num_episodes=args.num_eval_episodes
            )
            
            print(f"Evaluation: "
                  f"Reward = {eval_metrics['reward']:.4f}, "
                  f"Dice = {eval_metrics['dice']:.4f}, "
                  f"IoU = {eval_metrics['iou']:.4f}")
            
            # Save best model
            if eval_metrics['reward'] > best_reward:
                best_reward = eval_metrics['reward']
                agent.save(os.path.join(args.results_dir, 'best_agent.pt'))
                print(f"New best model saved with reward: {best_reward:.4f}")
        
        # Save model
        if (episode + 1) % args.save_interval == 0:
            agent.save(os.path.join(args.results_dir, f'agent_episode_{episode+1}.pt'))
    
    # Save final model
    agent.save(os.path.join(args.results_dir, 'final_agent.pt'))
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_dices)
    plt.title('Episode Dice Scores')
    plt.xlabel('Episode')
    plt.ylabel('Dice Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'training_curves.png'))

if __name__ == '__main__':
    args = parse_args()
    train(args) 