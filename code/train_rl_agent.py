import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
import json
from collections import defaultdict
import torch.nn.functional as F

from code.segmentation_env import SegmentationEnv, SegmentationEnvDataset
from code.enhanced_rl_agent import EnhancedPPOAgent
from code.models import UNet

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent for interactive segmentation')
    
    # Dataset parameters
    parser.add_argument('--image_dir', type=str, default='data/raw/Original',
                      help='Directory containing images')
    parser.add_argument('--mask_dir', type=str, default='data/raw/Ground Truth',
                      help='Directory containing ground truth masks')
    
    # Environment parameters
    parser.add_argument('--n_segments', type=int, default=100,
                      help='Number of superpixels to generate')
    parser.add_argument('--max_steps', type=int, default=20,
                      help='Maximum number of steps per episode')
    parser.add_argument('--reward_type', type=str, default='dice',
                      choices=['dice', 'iou'],
                      help='Type of reward to use')
    
    # Agent parameters
    parser.add_argument('--feature_extractor', type=str, default='cnn',
                      choices=['cnn', 'unet'],
                      help='Type of feature extractor to use')
    parser.add_argument('--unet_path', type=str, default='results/unet_model.pt',
                      help='Path to pretrained UNet model')
    parser.add_argument('--hidden_size', type=int, default=128,
                      help='Hidden size of agent')
    parser.add_argument('--lr', type=float, default=3e-4,
                      help='Learning rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--update_steps', type=int, default=2048,
                      help='Number of steps to collect before update')
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                      help='PPO clip ratio')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                      help='GAE lambda parameter')
    parser.add_argument('--value_coef', type=float, default=0.5,
                      help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                      help='Entropy loss coefficient')
    
    # Output parameters
    parser.add_argument('--results_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--save_interval', type=int, default=10,
                      help='Interval to save agent')
    parser.add_argument('--eval_interval', type=int, default=5,
                      help='Interval to evaluate agent')
    parser.add_argument('--n_eval_episodes', type=int, default=10,
                      help='Number of episodes to evaluate')
    parser.add_argument('--viz_path', type=str, default='data/example_images/rl_training',
                      help='Path to save visualizations')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    
    # Seed
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

def create_agent(args, env):
    """Create agent based on arguments"""
    # Get observation shape
    obs_shape = env.observation_space.shape
    
    # Get number of actions
    n_actions = env.action_space.n
    
    # Create feature extractor
    if args.feature_extractor == 'unet':
        # Load pretrained UNet
        unet = UNet(n_channels=3, n_classes=1)
        unet.load_state_dict(torch.load(args.unet_path, map_location=args.device))
        for param in unet.parameters():
            param.requires_grad = False
        unet.eval()
        
        feature_extractor = unet
        feature_extractor_type = 'unet'
    else:
        feature_extractor = None
        feature_extractor_type = 'cnn'
    
    # Create agent
    agent = EnhancedPPOAgent(
        obs_shape=obs_shape,
        action_space=n_actions,
        hidden_size=args.hidden_size,
        feature_extractor=feature_extractor,
        feature_extractor_type=feature_extractor_type,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        device=torch.device(args.device)
    )
    
    return agent

def collect_rollout(env, agent, render=False, render_path=None):
    """Collect a single rollout (episode) from the environment
    
    Args:
        env: Environment to collect from
        agent: Agent to use for action selection
        render: Whether to render the environment
        render_path: Path to save renders
        
    Returns:
        total_reward: Total reward for the episode
        episode_length: Length of the episode
        metrics: Dictionary of metrics
    """
    # Reset environment
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(agent.device)
    
    # Initialize variables
    done = False
    total_reward = 0
    actions = []
    episode_length = 0
    
    # Run episode
    while not done and episode_length < env.max_steps:
        # Select action
        action = agent.select_action(obs)
        actions.append(action.item())
        
        # Take step
        next_obs, reward, done, info = env.step(action.item())
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(agent.device)
        
        # Store experience
        agent.store_experience(
            obs=obs,
            action=action,
            reward=torch.tensor(reward, dtype=torch.float32).to(agent.device),
            next_obs=next_obs,
            done=torch.tensor(done, dtype=torch.float32).to(agent.device)
        )
        
        # Update variables
        obs = next_obs
        total_reward += reward
        episode_length += 1
    
    # Render final state if requested
    if render and render_path is not None:
        os.makedirs(os.path.dirname(render_path), exist_ok=True)
        env.render_interaction_sequence(actions, filename=render_path)
    
    return total_reward, episode_length, info

def evaluate_agent(agent, env_dataset, n_episodes, device, render=False, render_path=None):
    """Evaluate agent on a dataset of environments
    
    Args:
        agent: Agent to evaluate
        env_dataset: Dataset of environments
        n_episodes: Number of episodes to evaluate
        device: Device to use
        render: Whether to render the environment
        render_path: Path to save renders
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Initialize metrics
    metrics = defaultdict(list)
    
    # Sample environments
    if n_episodes > len(env_dataset):
        print(f"Warning: n_episodes ({n_episodes}) > len(env_dataset) ({len(env_dataset)})")
        n_episodes = len(env_dataset)
    
    indices = np.random.choice(len(env_dataset), n_episodes, replace=False)
    
    # Evaluate on each environment
    for i, idx in enumerate(indices):
        env = env_dataset[idx]
        
        # Collect rollout
        episode_render_path = None
        if render and render_path is not None:
            os.makedirs(render_path, exist_ok=True)
            episode_render_path = os.path.join(render_path, f"eval_episode_{i}.png")
        
        reward, length, info = collect_rollout(env, agent, render, episode_render_path)
        
        # Store metrics
        metrics['reward'].append(reward)
        metrics['length'].append(length)
        metrics['dice'].append(info['dice'])
        metrics['iou'].append(info['iou'])
    
    # Compute mean metrics
    for key in metrics:
        metrics[key] = float(np.mean(metrics[key]))
    
    return metrics

def train(args):
    """Train RL agent
    
    Args:
        args: Arguments
    """
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.viz_path, exist_ok=True)
    
    # Create environment dataset
    device = torch.device(args.device)
    env_dataset = SegmentationEnvDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        n_segments=args.n_segments,
        max_steps=args.max_steps,
        reward_type=args.reward_type,
        device=device
    )
    
    # Split dataset into train and val
    dataset_size = len(env_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, dataset_size))
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # Create agent
    env = env_dataset[0]  # Use first environment to initialize agent
    agent = create_agent(args, env)
    
    # Initialize training variables
    total_steps = 0
    episode_count = 0
    best_val_reward = -float('inf')
    
    # Training history
    history = {
        'train_rewards': [],
        'train_lengths': [],
        'train_dices': [],
        'train_ious': [],
        'val_rewards': [],
        'val_lengths': [],
        'val_dices': [],
        'val_ious': [],
        'epochs': [],
        'total_steps': []
    }
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Shuffle train indices
        random.shuffle(train_indices)
        
        # Training phase
        agent.train()
        train_metrics = defaultdict(list)
        pbar = tqdm(train_indices)
        
        for idx in pbar:
            # Get environment
            env = env_dataset[idx]
            
            # Render path for last episode
            render_path = None
            if episode_count % 50 == 0:
                render_path = os.path.join(args.viz_path, f"train_episode_{episode_count}.png")
            
            # Collect rollout
            reward, length, info = collect_rollout(env, agent, render=(render_path is not None), render_path=render_path)
            
            # Update metrics
            train_metrics['reward'].append(reward)
            train_metrics['length'].append(length)
            train_metrics['dice'].append(info['dice'])
            train_metrics['iou'].append(info['iou'])
            
            # Update progress bar
            pbar.set_description(f"Reward: {reward:.3f}, Dice: {info['dice']:.3f}, IoU: {info['iou']:.3f}")
            
            # Update agent
            if len(agent.experience_buffer) >= args.update_steps:
                loss_info = agent.update()
                agent.experience_buffer.clear()
                
                # Update progress bar
                pbar.set_description(
                    f"Reward: {reward:.3f}, Dice: {info['dice']:.3f}, "
                    f"Loss: {loss_info['total_loss']:.3f}, "
                    f"Policy: {loss_info['policy_loss']:.3f}, "
                    f"Value: {loss_info['value_loss']:.3f}, "
                    f"Entropy: {loss_info['entropy_loss']:.3f}"
                )
            
            # Update counters
            total_steps += length
            episode_count += 1
        
        # Compute mean training metrics
        for key in train_metrics:
            train_metrics[key] = float(np.mean(train_metrics[key]))
        
        # Evaluation phase
        if (epoch + 1) % args.eval_interval == 0:
            agent.eval()
            
            # Evaluate on validation set
            val_metrics = evaluate_agent(
                agent=agent,
                env_dataset=env_dataset,
                n_episodes=args.n_eval_episodes,
                device=device,
                render=True,
                render_path=os.path.join(args.viz_path, f"val_epoch_{epoch+1}")
            )
            
            # Print metrics
            print(f"Epoch {epoch+1}: " + 
                  f"Train Reward: {train_metrics['reward']:.3f}, " +
                  f"Train Dice: {train_metrics['dice']:.3f}, " +
                  f"Val Reward: {val_metrics['reward']:.3f}, " +
                  f"Val Dice: {val_metrics['dice']:.3f}")
            
            # Save best model
            if val_metrics['reward'] > best_val_reward:
                best_val_reward = val_metrics['reward']
                agent.save(os.path.join(args.results_dir, "best_rl_agent.pt"))
                print(f"New best model saved with validation reward: {best_val_reward:.3f}")
            
            # Update history
            history['val_rewards'].append(val_metrics['reward'])
            history['val_lengths'].append(val_metrics['length'])
            history['val_dices'].append(val_metrics['dice'])
            history['val_ious'].append(val_metrics['iou'])
        else:
            print(f"Epoch {epoch+1}: " + 
                  f"Train Reward: {train_metrics['reward']:.3f}, " +
                  f"Train Dice: {train_metrics['dice']:.3f}")
        
        # Update history
        history['train_rewards'].append(train_metrics['reward'])
        history['train_lengths'].append(train_metrics['length'])
        history['train_dices'].append(train_metrics['dice'])
        history['train_ious'].append(train_metrics['iou'])
        history['epochs'].append(epoch + 1)
        history['total_steps'].append(total_steps)
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            agent.save(os.path.join(args.results_dir, f"rl_agent_epoch_{epoch+1}.pt"))
            
            # Save history
            with open(os.path.join(args.results_dir, "rl_training_history.json"), 'w') as f:
                json.dump(history, f)
            
            # Plot training curves
            plot_training_curves(history, os.path.join(args.results_dir, "rl_training_curves.png"))
    
    # Save final model
    agent.save(os.path.join(args.results_dir, "final_rl_agent.pt"))
    
    # Save history
    with open(os.path.join(args.results_dir, "rl_training_history.json"), 'w') as f:
        json.dump(history, f)
    
    # Plot training curves
    plot_training_curves(history, os.path.join(args.results_dir, "rl_training_curves.png"))
    
    print("Training completed!")

def plot_training_curves(history, save_path):
    """Plot training curves
    
    Args:
        history: Dictionary of training history
        save_path: Path to save the plot
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    axs[0, 0].plot(history['epochs'], history['train_rewards'], label='Train')
    if len(history['val_rewards']) > 0:
        val_epochs = history['epochs'][::args.eval_interval]
        axs[0, 0].plot(val_epochs, history['val_rewards'], label='Val')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].set_title('Rewards')
    axs[0, 0].legend()
    
    # Plot Dice scores
    axs[0, 1].plot(history['epochs'], history['train_dices'], label='Train')
    if len(history['val_dices']) > 0:
        axs[0, 1].plot(val_epochs, history['val_dices'], label='Val')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Dice')
    axs[0, 1].set_title('Dice Scores')
    axs[0, 1].legend()
    
    # Plot IoU scores
    axs[1, 0].plot(history['epochs'], history['train_ious'], label='Train')
    if len(history['val_ious']) > 0:
        axs[1, 0].plot(val_epochs, history['val_ious'], label='Val')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('IoU')
    axs[1, 0].set_title('IoU Scores')
    axs[1, 0].legend()
    
    # Plot episode lengths
    axs[1, 1].plot(history['epochs'], history['train_lengths'], label='Train')
    if len(history['val_lengths']) > 0:
        axs[1, 1].plot(val_epochs, history['val_lengths'], label='Val')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Length')
    axs[1, 1].set_title('Episode Lengths')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    train(args) 