import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time
import wandb
import random
from collections import deque
from PIL import Image
import cv2
from glob import glob
import logging
from datetime import datetime

# Add code directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.data_utils import PolypDataset
from code.enhanced_rl_agent import EnhancedPPOAgent, PolicyNetwork, ValueNetwork, EnhancedFeatureExtractor
from code.enhanced_rl_env import EnhancedPolypSegmentationEnv, create_env_factory
from code.models import UNet
from code.utils import dice_coefficient, iou_score, save_image, plot_losses, set_seed
from code.enhanced_segmentation_env import create_segmentation_env

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train enhanced RL agent for interactive polyp segmentation')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Path to data directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Path to results directory')
    parser.add_argument('--save_dir', type=str, default='models', help='Path to save models')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes for training')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--curriculum_steps', type=int, default=200, help='Number of episodes before increasing difficulty')
    
    # Environment parameters
    parser.add_argument('--pointer_radius', type=int, default=5, help='Radius of the pointer')
    parser.add_argument('--expansion_factor', type=float, default=1.05, help='Factor for expansion action')
    parser.add_argument('--shrink_factor', type=float, default=0.95, help='Factor for shrink action')
    parser.add_argument('--move_step', type=int, default=5, help='Step size for movement actions')
    
    # Model parameters
    parser.add_argument('--feature_extractor', type=str, default='unet', help='Feature extractor type (unet or cnn)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for policy and value networks')
    parser.add_argument('--load_unet', type=str, default=None, help='Path to pre-trained UNet model')
    parser.add_argument('--load_agent', type=str, default=None, help='Path to pre-trained RL agent')
    
    # Logging parameters
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging')
    parser.add_argument('--eval_interval', type=int, default=100, help='Interval for evaluation')
    parser.add_argument('--save_interval', type=int, default=100, help='Interval for saving models')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--visualize', action='store_true', help='Visualize episodes during training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    return args

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

def load_data(data_dir, batch_size):
    """Load data from directory"""
    # Create dataset
    dataset = PolypDataset(data_dir, split='train')
    valid_dataset = PolypDataset(data_dir, split='val')
    test_dataset = PolypDataset(data_dir, split='test')
    
    # Create data loaders
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader, valid_dataloader, test_dataloader, dataset, valid_dataset, test_dataset

def load_or_train_unet(args, dataloader, valid_dataloader):
    """Load pre-trained UNet or train a new one"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.load_unet:
        print(f"Loading pre-trained UNet from {args.load_unet}")
        unet = UNet(n_channels=3, n_classes=1)
        unet.load_state_dict(torch.load(args.load_unet, map_location=device))
        unet = unet.to(device)
        return unet
    
    print("Training UNet...")
    unet = UNet(n_channels=3, n_classes=1)
    unet = unet.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(unet.parameters(), lr=args.learning_rate)
    
    # Train UNet for 10 epochs
    num_epochs = 10
    best_valid_dice = 0.0
    best_model = None
    
    for epoch in range(num_epochs):
        # Training
        unet.train()
        train_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = unet(images)
            loss = criterion(outputs, masks)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(dataloader.dataset)
        
        # Validation
        unet.eval()
        valid_loss = 0.0
        valid_dice = 0.0
        valid_iou = 0.0
        
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Forward pass
                outputs = unet(images)
                loss = criterion(outputs, masks)
                
                # Convert outputs to binary masks
                preds = torch.sigmoid(outputs) > 0.5
                
                # Calculate metrics
                dice = dice_coefficient(preds.cpu().numpy(), masks.cpu().numpy())
                iou = iou_score(preds.cpu().numpy(), masks.cpu().numpy())
                
                valid_loss += loss.item() * images.size(0)
                valid_dice += dice * images.size(0)
                valid_iou += iou * images.size(0)
        
        valid_loss /= len(valid_dataloader.dataset)
        valid_dice /= len(valid_dataloader.dataset)
        valid_iou /= len(valid_dataloader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, "
              f"Valid Dice: {valid_dice:.4f}, "
              f"Valid IoU: {valid_iou:.4f}")
        
        # Save best model
        if valid_dice > best_valid_dice:
            best_valid_dice = valid_dice
            best_model = unet.state_dict()
    
    # Save best model
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(best_model, os.path.join(args.save_dir, 'unet.pth'))
    print(f"Best UNet model saved with Dice: {best_valid_dice:.4f}")
    
    # Load best model
    unet.load_state_dict(best_model)
    
    return unet

def create_rl_agent(args, unet=None):
    """Create RL agent"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create feature extractor
    if args.feature_extractor == 'unet' and unet is not None:
        # Use pre-trained UNet as feature extractor
        feature_extractor = EnhancedFeatureExtractor(unet.encoder, device=device)
    else:
        # Create CNN feature extractor
        feature_extractor = EnhancedFeatureExtractor(None, device=device, use_cnn=True)
    
    # Create policy and value networks
    input_dim = feature_extractor.output_dim
    policy_network = PolicyNetwork(input_dim, args.hidden_dim)
    value_network = ValueNetwork(input_dim, args.hidden_dim)
    
    # Create PPO agent
    agent = EnhancedPPOAgent(
        feature_extractor=feature_extractor,
        policy_network=policy_network,
        value_network=value_network,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        device=device
    )
    
    # Load pre-trained agent if specified
    if args.load_agent:
        print(f"Loading pre-trained agent from {args.load_agent}")
        agent.load(args.load_agent)
    
    return agent

def train_rl_agent(args, agent, train_dataset, valid_dataset):
    """Train RL agent"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create environment factory
    train_images = [sample['image'].numpy() for sample in train_dataset]
    train_masks = [sample['mask'].numpy() for sample in train_dataset]
    
    valid_images = [sample['image'].numpy() for sample in valid_dataset]
    valid_masks = [sample['mask'].numpy() for sample in valid_dataset]
    
    # Create environment with base parameters
    env_kwargs = {
        'max_steps': args.max_steps,
        'pointer_radius': args.pointer_radius,
        'expansion_factor': args.expansion_factor,
        'shrink_factor': args.shrink_factor,
        'move_step': args.move_step,
        'device': device
    }
    
    train_env_factory = create_env_factory(train_images, train_masks, **env_kwargs)
    valid_env_factory = create_env_factory(valid_images, valid_masks, **env_kwargs)
    
    # Setup logging
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.wandb:
        wandb.init(project="interactive-polyp-segmentation", name="enhanced-rl-agent")
        wandb.config.update(args)
    
    # Initialize training variables
    best_valid_dice = 0.0
    episode_rewards = []
    episode_dices = []
    episode_ious = []
    episode_steps = []
    
    # Curriculum learning difficulty levels
    difficulty_levels = [
        {'pointer_radius': 8, 'move_step': 8},  # Easy
        {'pointer_radius': 6, 'move_step': 6},  # Medium
        {'pointer_radius': 5, 'move_step': 5},  # Hard
        {'pointer_radius': 4, 'move_step': 4},  # Harder
        {'pointer_radius': 3, 'move_step': 3}   # Hardest
    ]
    current_difficulty = 0
    
    # Create buffer for experience replay
    replay_buffer = deque(maxlen=1000)
    
    # Training loop
    for episode in range(args.num_episodes):
        # Update difficulty if needed (curriculum learning)
        if episode > 0 and episode % args.curriculum_steps == 0 and current_difficulty < len(difficulty_levels) - 1:
            current_difficulty += 1
            print(f"Increasing difficulty to level {current_difficulty+1}/{len(difficulty_levels)}")
            # Update environment parameters
            env_kwargs.update(difficulty_levels[current_difficulty])
            train_env_factory = create_env_factory(train_images, train_masks, **env_kwargs)
        
        # Create environment
        env = train_env_factory()
        
        # Reset environment and agent
        obs = env.reset()
        done = False
        episode_reward = 0.0
        
        # Collect episode experience
        while not done:
            # Select action
            action, action_log_prob, value = agent.select_action(obs)
            
            # Take action in environment
            next_obs, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_experience(obs, action, reward, done, action_log_prob, value)
            
            # Visualize episode if requested
            if args.visualize and episode % args.log_interval == 0:
                env.render(show=True)
            
            # Update for next iteration
            obs = next_obs
            episode_reward += reward
        
        # Store episode in replay buffer
        replay_buffer.append(agent.get_experiences())
        
        # Update agent after each episode
        loss_info = agent.update()
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_dices.append(info['dice'])
        episode_ious.append(info['iou'])
        episode_steps.append(info['steps'])
        
        # Log episode
        if (episode + 1) % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval:])
            avg_dice = np.mean(episode_dices[-args.log_interval:])
            avg_iou = np.mean(episode_ious[-args.log_interval:])
            avg_steps = np.mean(episode_steps[-args.log_interval:])
            
            print(f"Episode {episode+1}/{args.num_episodes}, "
                  f"Avg Reward: {avg_reward:.4f}, "
                  f"Avg Dice: {avg_dice:.4f}, "
                  f"Avg IoU: {avg_iou:.4f}, "
                  f"Avg Steps: {avg_steps:.1f}")
            
            if args.wandb:
                wandb.log({
                    'episode': episode + 1,
                    'train/reward': avg_reward,
                    'train/dice': avg_dice,
                    'train/iou': avg_iou,
                    'train/steps': avg_steps,
                    'train/policy_loss': loss_info['policy_loss'],
                    'train/value_loss': loss_info['value_loss'],
                    'train/entropy_loss': loss_info['entropy_loss'],
                    'train/total_loss': loss_info['total_loss'],
                    'train/learning_rate': agent.optimizer.param_groups[0]['lr'],
                    'train/difficulty_level': current_difficulty + 1
                })
        
        # Evaluate agent
        if (episode + 1) % args.eval_interval == 0:
            valid_dices, valid_ious, valid_steps, valid_rewards, valid_images_list = evaluate_agent(
                agent, valid_env_factory, num_episodes=min(50, len(valid_dataset)), visualize=False
            )
            
            valid_dice = np.mean(valid_dices)
            valid_iou = np.mean(valid_ious)
            valid_steps = np.mean(valid_steps)
            valid_reward = np.mean(valid_rewards)
            
            print(f"Validation - "
                  f"Dice: {valid_dice:.4f}, "
                  f"IoU: {valid_iou:.4f}, "
                  f"Steps: {valid_steps:.1f}, "
                  f"Reward: {valid_reward:.4f}")
            
            if args.wandb:
                wandb.log({
                    'episode': episode + 1,
                    'valid/dice': valid_dice,
                    'valid/iou': valid_iou,
                    'valid/steps': valid_steps,
                    'valid/reward': valid_reward
                })
                
                # Log sample images
                if len(valid_images_list) > 0:
                    wandb.log({"valid/samples": [wandb.Image(img) for img in valid_images_list[:5]]})
            
            # Save best model
            if valid_dice > best_valid_dice:
                best_valid_dice = valid_dice
                agent.save(os.path.join(args.save_dir, 'best_rl_agent.pt'))
                print(f"New best model saved with Dice: {best_valid_dice:.4f}")
        
        # Save model periodically
        if (episode + 1) % args.save_interval == 0:
            agent.save(os.path.join(args.save_dir, f'rl_agent_episode_{episode+1}.pt'))
            
            # Save training history
            history = {
                'rewards': episode_rewards,
                'dices': episode_dices,
                'ious': episode_ious,
                'steps': episode_steps
            }
            
            with open(os.path.join(args.results_dir, 'training_history.pkl'), 'wb') as f:
                pickle.dump(history, f)
            
            # Plot training history
            plot_training_history(history, os.path.join(args.results_dir, 'training_history.png'))
    
    # Save final model
    agent.save(os.path.join(args.save_dir, 'final_rl_agent.pt'))
    
    # Save training history
    history = {
        'rewards': episode_rewards,
        'dices': episode_dices,
        'ious': episode_ious,
        'steps': episode_steps
    }
    
    with open(os.path.join(args.results_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # Plot training history
    plot_training_history(history, os.path.join(args.results_dir, 'training_history.png'))
    
    return agent, history

def evaluate_agent(agent, env_factory, num_episodes=50, visualize=False):
    """Evaluate agent on multiple episodes"""
    dices = []
    ious = []
    steps = []
    rewards = []
    images_list = []
    
    for i in range(num_episodes):
        env = env_factory(i)  # Fixed index for reproducibility
        obs = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            # Select action
            action, _, _ = agent.select_action(obs)
            
            # Take action in environment
            next_obs, reward, done, info = env.step(action)
            
            # Update for next iteration
            obs = next_obs
            episode_reward += reward
        
        # Store episode statistics
        dices.append(info['dice'])
        ious.append(info['iou'])
        steps.append(info['steps'])
        rewards.append(episode_reward)
        
        # Store visualization
        if visualize:
            render_img = env.render(show=False)
            images_list.append(render_img)
    
    return dices, ious, steps, rewards, images_list

def plot_training_history(history, save_path):
    """Plot training history"""
    # Create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot rewards
    axs[0, 0].plot(history['rewards'])
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    
    # Plot Dice coefficient
    axs[0, 1].plot(history['dices'])
    axs[0, 1].set_title('Episode Dice Coefficient')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Dice')
    
    # Plot IoU
    axs[1, 0].plot(history['ious'])
    axs[1, 0].set_title('Episode IoU')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('IoU')
    
    # Plot steps
    axs[1, 1].plot(history['steps'])
    axs[1, 1].set_title('Episode Steps')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Steps')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_agent_performance(agent, env_factory, num_episodes=5, save_dir='results'):
    """Visualize agent performance"""
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_episodes):
        env = env_factory(i)  # Fixed index for reproducibility
        obs = env.reset()
        done = False
        step = 0
        
        # Prepare figure for animation
        fig, ax = plt.subplots(figsize=(10, 8))
        frames = []
        
        while not done:
            # Select action
            action, _, _ = agent.select_action(obs)
            
            # Take action in environment
            next_obs, reward, done, info = env.step(action)
            
            # Render environment
            render_img = env.render(show=False)
            frames.append(render_img)
            
            # Update for next iteration
            obs = next_obs
            step += 1
        
        # Save frames as video or images
        for j, frame in enumerate(frames):
            plt.figure(figsize=(10, 8))
            plt.imshow(frame)
            plt.title(f"Step {j+1}/{len(frames)}, Dice: {info['dice']:.4f}, IoU: {info['iou']:.4f}")
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f'episode_{i+1}_step_{j+1}.png'))
            plt.close()
        
        # Create a grid of images showing progression
        num_show = min(len(frames), 5)
        step_indices = np.linspace(0, len(frames)-1, num_show, dtype=int)
        
        plt.figure(figsize=(15, 8))
        for j, idx in enumerate(step_indices):
            plt.subplot(1, num_show, j+1)
            plt.imshow(frames[idx])
            plt.title(f"Step {idx+1}")
            plt.axis('off')
        
        plt.suptitle(f"Episode {i+1}, Final Dice: {info['dice']:.4f}, IoU: {info['iou']:.4f}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'episode_{i+1}_progression.png'))
        plt.close()

def load_dataset(image_dir, mask_dir, test_split=0.2, val_split=0.2):
    """Load dataset from directories
    
    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        test_split: Fraction of data to use for testing
        val_split: Fraction of data to use for validation
    
    Returns:
        Tuple of (train_images, train_masks, val_images, val_masks, test_images, test_masks)
    """
    # Get image and mask paths
    image_paths = sorted(glob(os.path.join(image_dir, "*.tif")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.tif")))
    
    if len(image_paths) == 0:
        image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
    if len(mask_paths) == 0:
        mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
    
    logger.info(f"Found {len(image_paths)} images")
    
    if len(image_paths) != len(mask_paths):
        raise ValueError(f"Number of images ({len(image_paths)}) doesn't match number of masks ({len(mask_paths)})")
    
    # Create list of indices
    indices = list(range(len(image_paths)))
    random.shuffle(indices)
    
    # Split into train, val, test
    test_size = int(len(indices) * test_split)
    val_size = int(len(indices) * val_split)
    train_indices = indices[test_size + val_size:]
    val_indices = indices[test_size:test_size + val_size]
    test_indices = indices[:test_size]
    
    # Load images and masks
    train_images = []
    train_masks = []
    val_images = []
    val_masks = []
    test_images = []
    test_masks = []
    
    logger.info("Loading training images and masks...")
    for idx in tqdm(train_indices):
        image = np.array(Image.open(image_paths[idx]))
        mask = np.array(Image.open(mask_paths[idx]))
        train_images.append(image)
        train_masks.append(mask)
    
    logger.info("Loading validation images and masks...")
    for idx in tqdm(val_indices):
        image = np.array(Image.open(image_paths[idx]))
        mask = np.array(Image.open(mask_paths[idx]))
        val_images.append(image)
        val_masks.append(mask)
    
    logger.info("Loading test images and masks...")
    for idx in tqdm(test_indices):
        image = np.array(Image.open(image_paths[idx]))
        mask = np.array(Image.open(mask_paths[idx]))
        test_images.append(image)
        test_masks.append(mask)
    
    logger.info(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    return (train_images, train_masks, val_images, val_masks, test_images, test_masks)

def train_ppo_agent(env, val_env, agent, num_episodes, save_dir, log_interval=10, save_interval=100):
    """Train PPO agent
    
    Args:
        env: Training environment
        val_env: Validation environment
        agent: PPO agent
        num_episodes: Number of episodes to train for
        save_dir: Directory to save model and visualizations
        log_interval: How often to log information
        save_interval: How often to save model
    
    Returns:
        Dictionary of training history
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "visualizations"), exist_ok=True)
    
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'dice_scores': [],
        'iou_scores': [],
        'val_dice_scores': [],
        'val_iou_scores': [],
        'losses': [],
        'value_losses': [],
        'policy_losses': [],
        'entropies': []
    }
    
    # Initialize validation tracking
    best_val_dice = 0.0
    
    # Enable visualization for validation environment
    val_env.enable_visualization(True)
    
    # Start training
    logger.info(f"Starting training for {num_episodes} episodes")
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        state, info = env.reset()
        
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        # Update agent
        metrics = agent.update()
        
        # Store episode information
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(env.steps)
        history['dice_scores'].append(info['dice'])
        history['iou_scores'].append(info['iou'])
        
        # Store loss information
        if metrics is not None:
            history['losses'].append(metrics['loss'])
            history['value_losses'].append(metrics['value_loss'])
            history['policy_losses'].append(metrics['policy_loss'])
            history['entropies'].append(metrics['entropy'])
        
        # Evaluate on validation set
        if episode % log_interval == 0:
            val_dice, val_iou = evaluate_agent(val_env, agent, num_episodes=5)
            history['val_dice_scores'].append(val_dice)
            history['val_iou_scores'].append(val_iou)
            
            logger.info(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {episode_reward:.4f} | "
                      f"Dice: {info['dice']:.4f} | "
                      f"IoU: {info['iou']:.4f} | "
                      f"Val Dice: {val_dice:.4f} | "
                      f"Val IoU: {val_iou:.4f}")
            
            # Save visualization
            vis_path = os.path.join(save_dir, "visualizations", f"episode_{episode}.png")
            val_env.render_history(save_path=vis_path)
            
            # Save model if it's the best so far
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                agent.save(os.path.join(save_dir, "checkpoints", "best_model.pt"))
                logger.info(f"Saved new best model with validation Dice {best_val_dice:.4f}")
        
        # Save model at regular intervals
        if episode % save_interval == 0:
            agent.save(os.path.join(save_dir, "checkpoints", f"model_episode_{episode}.pt"))
            
            # Save training curves
            plot_losses(
                history['episode_rewards'], 
                history['dice_scores'], 
                history['val_dice_scores'] if history['val_dice_scores'] else None,
                os.path.join(save_dir, f"training_curve_episode_{episode}.png")
            )
    
    # Save final model
    agent.save(os.path.join(save_dir, "checkpoints", "final_model.pt"))
    
    return history

def interactive_segmentation_demo(env, agent, save_dir, num_samples=5):
    """Run interactive segmentation demo
    
    Args:
        env: Environment to run demo on
        agent: Agent to use for segmentation
        save_dir: Directory to save visualizations
        num_samples: Number of samples to process
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Enable visualization
    env.enable_visualization(True)
    
    # Process samples
    for sample_idx in range(num_samples):
        logger.info(f"Processing sample {sample_idx + 1}/{num_samples}")
        
        # Reset environment with specific sample
        state, info = env.reset(image_idx=sample_idx)
        
        done = False
        step = 0
        
        while not done:
            # Select action
            action = agent.select_action(state, deterministic=True)
            
            # Take step in environment
            state, reward, done, info = env.step(action)
            
            step += 1
            
            # Log progress
            if step % 10 == 0:
                logger.info(f"Step {step} | Dice: {info['dice']:.4f} | IoU: {info['iou']:.4f}")
        
        # Save visualization
        vis_path = os.path.join(save_dir, f"sample_{sample_idx}_segmentation.png")
        env.render_history(save_path=vis_path)
        
        logger.info(f"Final Dice: {info['dice']:.4f} | Final IoU: {info['iou']:.4f}")

def main(args):
    """Main function
    
    Args:
        args: Command-line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    train_images, train_masks, val_images, val_masks, test_images, test_masks = load_dataset(
        args.image_dir, args.mask_dir, args.test_split, args.val_split
    )
    
    # Create environments
    logger.info("Creating environments...")
    train_env = create_segmentation_env(
        train_images, train_masks, 
        image_size=(args.image_size, args.image_size), 
        difficulty=args.difficulty
    )
    
    val_env = create_segmentation_env(
        val_images, val_masks, 
        image_size=(args.image_size, args.image_size), 
        difficulty=args.difficulty
    )
    
    test_env = create_segmentation_env(
        test_images, test_masks, 
        image_size=(args.image_size, args.image_size), 
        difficulty=args.difficulty
    )
    
    # Create or load UNet for feature extraction
    if args.use_unet_extractor:
        logger.info("Creating UNet feature extractor...")
        feature_extractor = UNet(n_channels=3, n_classes=1)
        
        if args.unet_pretrained:
            logger.info(f"Loading pretrained UNet from {args.unet_pretrained}")
            feature_extractor.load_state_dict(torch.load(args.unet_pretrained, map_location=device))
        
        # Only use the encoder part for feature extraction
        feature_extractor_mode = "unet"
    else:
        logger.info("Using CNN feature extractor...")
        feature_extractor = None
        feature_extractor_mode = "cnn"
    
    # Create PPO agent
    logger.info("Creating PPO agent...")
    agent = EnhancedPPOAgent(
        state_dim=(7, args.image_size, args.image_size),
        action_dim=train_env.action_space.n,
        feature_extractor=feature_extractor,
        feature_extractor_mode=feature_extractor_mode,
        device=device,
        lr=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_param=args.clip_param,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size
    )
    
    # Load agent if specified
    if args.load_agent:
        logger.info(f"Loading agent from {args.load_agent}")
        agent.load(args.load_agent)
    
    # Set up timestamp for saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"ppo_agent_{timestamp}")
    
    if args.mode == 'train':
        # Train agent
        logger.info("Starting training...")
        history = train_ppo_agent(
            train_env, val_env, agent, 
            num_episodes=args.num_episodes,
            save_dir=save_dir,
            log_interval=args.log_interval,
            save_interval=args.save_interval
        )
        
        # Plot training curves
        logger.info("Plotting training curves...")
        plot_losses(
            history['episode_rewards'], 
            history['dice_scores'], 
            history['val_dice_scores'] if history['val_dice_scores'] else None,
            os.path.join(save_dir, "final_training_curve.png")
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_dice, test_iou = evaluate_agent(test_env, agent, num_episodes=len(test_images))
        logger.info(f"Test Dice: {test_dice:.4f} | Test IoU: {test_iou:.4f}")
        
        # Save evaluation results
        with open(os.path.join(save_dir, "test_results.txt"), "w") as f:
            f.write(f"Test Dice: {test_dice:.4f}\n")
            f.write(f"Test IoU: {test_iou:.4f}\n")
    
    elif args.mode == 'demo':
        # Run interactive segmentation demo
        logger.info("Running interactive segmentation demo...")
        interactive_segmentation_demo(
            test_env, agent, 
            save_dir=os.path.join(save_dir, "demo"),
            num_samples=min(args.num_demo_samples, len(test_images))
        )
    
    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Polyp Segmentation with RL")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "demo"],
                        help="Mode to run: 'train' or 'demo'")
    parser.add_argument("--image_dir", type=str, default="data/raw/Original",
                        help="Directory containing images")
    parser.add_argument("--mask_dir", type=str, default="data/raw/Ground Truth",
                        help="Directory containing masks")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--load_agent", type=str, default=None,
                        help="Path to load agent from")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Size to resize images to")
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of episodes to train for")
    parser.add_argument("--test_split", type=float, default=0.15,
                        help="Fraction of data to use for testing")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Fraction of data to use for validation")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"],
                        help="Difficulty level of environment")
    parser.add_argument("--learning_rate", type=float, default=0.0003,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda parameter")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help="PPO clip parameter")
    parser.add_argument("--value_loss_coef", type=float, default=0.5,
                        help="Value loss coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="Maximum gradient norm")
    parser.add_argument("--ppo_epochs", type=int, default=10,
                        help="Number of epochs for PPO updates")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for PPO updates")
    parser.add_argument("--buffer_size", type=int, default=2048,
                        help="Size of experience buffer")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Interval between logging information")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Interval between saving model")
    parser.add_argument("--num_demo_samples", type=int, default=5,
                        help="Number of samples to process in demo mode")
    parser.add_argument("--use_unet_extractor", action="store_true",
                        help="Use UNet encoder as feature extractor")
    parser.add_argument("--unet_pretrained", type=str, default=None,
                        help="Path to pretrained UNet model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA")
    
    args = parser.parse_args()
    main(args) 