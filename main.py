import os
import sys
import argparse
import torch
import numpy as np
import random
from pathlib import Path
import cv2
import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import pickle
import json
from sklearn.model_selection import train_test_split

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from src.data_utils import PolypDataset, get_data_loaders, visualize_sample
from src.unet_model import UNet, train_unet
from src.rl_environment import PolypSegmentationEnv, PolypFeatureExtractor
from src.rl_agent import PPOAgent
from src.inter_rl import train_agent, SimpleRLAgent
from src.inter_rl import predict_mask, visualize_results

def setup_environment(seed=42):
    """Set up the environment (directories, random seeds, etc.)"""
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return device

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Reinforcement Learning for Interactive Polyp Segmentation')
    
    parser.add_argument('--mode', type=str, default='all', choices=['train_unet', 'train_rl', 'train_simple_rl', 'evaluate', 'all', 'compare_unet_simple_rl', 'train_and_evaluate'],
                        help='Mode of operation: train_unet (train U-Net only), train_rl (train PPO agent), train_simple_rl (train InteractiveRL agent and save history), evaluate (evaluate models), all (run all), compare_unet_simple_rl (compare UNet and SimpleRL), train_and_evaluate (train and evaluate)')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Directory containing the dataset')
    parser.add_argument('--unet_epochs', type=int, default=1000,
                        help='Number of epochs for training U-Net')
    parser.add_argument('--rl_episodes', type=int, default=1000,
                        help='Number of episodes for training RL agent')
    parser.add_argument('--simple_rl_episodes', type=int, default=1000,
                        help='Number of episodes for training SimpleRL agent')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=20,
                        help='Maximum steps per episode for RL')
    parser.add_argument('--eval_samples', type=int, default=50,
                        help='Number of samples to evaluate on')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Whether to save visualization images during evaluation')
    parser.add_argument('--eval_interval', type=int, default=50,
                        help='Interval for validation evaluation during InteractiveRL training')
    parser.add_argument('--num_eval_episodes', type=int, default=5,
                        help='Number of evaluation episodes during InteractiveRL training validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--split_ratio', type=str, default='0.7,0.15,0.15',
                        help='Split ratio for train/val/test, comma-separated (e.g., 0.7,0.15,0.15)')
    
    return parser.parse_args()

def get_improved_data_loaders(data_dir, batch_size, num_workers=2, seed=42, split_ratio=(0.7, 0.15, 0.15)):
    """Get improved data loaders with better shuffling and stratified splits
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
        split_ratio: Tuple of (train, val, test) ratios
        
    Returns:
        Dictionary containing data loaders for train, val, test
    """
    from src.data_utils import PolypDataset
    from torch.utils.data import DataLoader, Subset, Dataset
    import glob
    import os
    
    # Ensure split ratios sum to 1
    assert sum(split_ratio) == 1.0, "Split ratios must sum to 1"
    
    # Get all image files - adapted to the actual directory structure
    image_files = sorted(glob.glob(os.path.join(data_dir, 'Original', '*.jpg')))
    image_files += sorted(glob.glob(os.path.join(data_dir, 'Original', '*.png')))
    image_files += sorted(glob.glob(os.path.join(data_dir, 'Original', '*.tif')))  # Add .tif extension
    
    print(f"Found {len(image_files)} image files in {os.path.join(data_dir, 'Original')}")
    
    # Get corresponding mask files
    mask_files = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        # Try different mask extensions
        mask_extensions = ['.png', '.jpg', '.tif']
        mask_path = None
        for ext in mask_extensions:
            potential_path = os.path.join(data_dir, 'Ground Truth', f"{base_name}{ext}")
            if os.path.exists(potential_path):
                mask_path = potential_path
                break
        
        if mask_path:
            mask_files.append(mask_path)
        else:
            print(f"Warning: No mask found for {filename}")
    
    # Ensure we have masks for all images
    assert len(image_files) == len(mask_files), f"Found {len(image_files)} images but {len(mask_files)} masks"
    
    # Create a custom dataset class that accepts image and mask file paths
    class CustomPolypDataset(Dataset):
        def __init__(self, image_paths, mask_paths):
            self.image_paths = image_paths
            self.mask_paths = mask_paths
            
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            # Load image
            image_path = self.image_paths[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load mask
            mask_path = self.mask_paths[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Normalize image to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Convert mask to binary
            mask = (mask > 0).astype(np.float32)
            
            # Convert to PyTorch tensors
            image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            mask = torch.from_numpy(mask).unsqueeze(0)  # (H, W) -> (1, H, W)
            
            return {'image': image, 'mask': mask}
    
    # Create a complete dataset
    full_dataset = CustomPolypDataset(image_files, mask_files)
    
    print(f"Created dataset with {len(full_dataset)} samples")
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * split_ratio[0])
    val_size = int(dataset_size * split_ratio[1])
    test_size = dataset_size - train_size - val_size
    
    # Split indices
    indices = list(range(dataset_size))
    
    # Shuffle indices
    random.seed(seed)
    random.shuffle(indices)
    
    # Create train, validation, and test splits
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Split dataset: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Save the split indices for reproducibility
    split_info = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'seed': seed,
        'split_ratio': split_ratio
    }
    
    os.makedirs('data/splits', exist_ok=True)
    with open('data/splits/dataset_split.json', 'w') as f:
        # Convert indices to ints for JSON serialization
        split_info['train_indices'] = [int(i) for i in train_indices]
        split_info['val_indices'] = [int(i) for i in val_indices]
        split_info['test_indices'] = [int(i) for i in test_indices]
        split_info['split_ratio'] = list(split_ratio)
        json.dump(split_info, f, indent=2)
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    print(f"Split info saved to data/splits/dataset_split.json")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'full_dataset': full_dataset,
        'split_info': split_info
    }

def optimize_unet(args, device, data_loaders):
    """
    Train an optimized U-Net model with advanced techniques and return the model
    """
    print("Starting optimized U-Net training...")
    
    # Create directories for model checkpoints
    unet_model_dir = os.path.join("models", "unet_optimized")
    os.makedirs(unet_model_dir, exist_ok=True)
    
    # Use optimized U-Net model (lightweight filter configuration)
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    model = model.to(device)
    
    # Use optimized loss function and training process
    from src.unet_model import train_unet, DiceBCELoss
    
    # Execute optimized U-Net training (including early stopping and LR scheduling)
    # Set to 20 epochs for comparison
    epochs = 5000 if args.unet_epochs > 20 else args.unet_epochs
    
    print(f"Starting optimized U-Net training, planning to train for {epochs} epochs...")
    
    # Train the model with early stopping and LR scheduling
    best_model, train_history = train_unet(
        model=model,
        device=device,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        epochs=epochs,
        learning_rate=args.lr,
        save_checkpoint=True,
        amp=True,  # Use mixed precision training
        checkpoint_dir=unet_model_dir,
        patience=15  # Early stopping patience
    )
    
    return best_model, train_history

def train_unet_model(args, device, data_loaders):
    """Train and evaluate the U-Net model with standardized outputs"""
    print('Training U-Net model...')
    
    # Create timestamp for this training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model output directories
    unet_model_dir = os.path.join('models', f'unet_{timestamp}')
    os.makedirs(unet_model_dir, exist_ok=True)
    
    # Use optimized U-Net model (lightweight filter configuration)
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    model = model.to(device)
    
    # Use optimized loss function and training process
    from src.unet_model import train_unet, DiceBCELoss
    
    # Execute optimized U-Net training (including early stopping and LR scheduling)
    # Set to 20 epochs for comparison
    epochs = 5000 if args.unet_epochs > 20 else args.unet_epochs
    
    print(f"Starting optimized U-Net training, planning to train for {epochs} epochs...")
    
    # Use optimized training function
    history = train_unet(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        device=device,
        epochs=epochs,
        lr=args.lr,
        patience=10  # Use smaller patience value to adapt to 20-epoch training
    )
    
    # For compatibility, retain key names from existing code
    full_history = {
        'epochs': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_dice': history['val_dice'],
        'val_iou': history['val_iou'],
        'lr': [args.lr] * len(history['train_loss']),  # Simplified, not tracking LR changes
        'best_epoch': history.get('best_epoch', 0) + 1,  # Index starting from 0 converted to 1-based epoch
        'best_val_dice': max(history['val_dice']) if history['val_dice'] else 0.0,
        'best_val_iou': max(history['val_iou']) if history['val_iou'] else 0.0,
        'total_training_time': 0.0,  # Will be updated later
        'early_stopped': history.get('early_stopped', False)
    }
    
    # Save all training history to a single JSON file
    with open(os.path.join(unet_model_dir, 'training_history.json'), 'w') as f:
        # Create a JSON serializable copy
        json_serializable = {}
        for key, value in full_history.items():
            if isinstance(value, (list, dict)):
                if len(value) > 0 and isinstance(value[0], (np.integer, np.floating, np.ndarray)):
                    json_serializable[key] = [float(v) for v in value]
                else:
                    json_serializable[key] = value
            elif isinstance(value, (np.integer, np.floating)):
                json_serializable[key] = float(value)
            else:
                json_serializable[key] = value
        
        json.dump(json_serializable, f, indent=2)
    
    # Also save as a specific name so generate_academic_plots.py can find it
    with open(os.path.join(unet_model_dir, 'unet_training_history.json'), 'w') as f:
        json.dump(json_serializable, f, indent=2)
    
    # Save the model to the standard location
    torch.save(model.state_dict(), 'models/unet_model.pth')
    
    # Save the best model
    best_model_path = os.path.join(unet_model_dir, 'best_model.pth')
    torch.save({
        'epoch': full_history['best_epoch'],
        'model_state_dict': model.state_dict(),
        'loss': min(history['val_loss']) if history['val_loss'] else 0.0,
        'dice': full_history['best_val_dice'],
        'iou': full_history['best_val_iou']
    }, best_model_path)
    
    print(f"Optimized U-Net training completed, best model saved at epoch {full_history['best_epoch']}, validation Dice: {full_history['best_val_dice']:.4f}")
    
    # Calculate final validation performance
    model.eval()
    val_dice = 0
    val_iou = 0
    
    with torch.no_grad():
        for batch in data_loaders['val']:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            pred_masks = (outputs > 0.5).float()
            
            from src.unet_model import dice_coefficient, iou_score
            batch_dice = dice_coefficient(pred_masks, masks).item()
            batch_iou = iou_score(pred_masks, masks).item()
            
            val_dice += batch_dice
            val_iou += batch_iou
    
    val_dice /= len(data_loaders['val'])
    val_iou /= len(data_loaders['val'])
    
    # Display training results summary
    print(f"\n=============== Optimized U-Net training summary ===============")
    print(f"Training epochs: {len(history['train_loss'])}/{epochs}")
    print(f"Best model epoch: {full_history['best_epoch']}")
    print(f"Best validation Dice: {full_history['best_val_dice']:.4f}")
    print(f"Best validation IoU: {full_history['best_val_iou']:.4f}")
    print(f"Final validation Dice: {val_dice:.4f}")
    print(f"Final validation IoU: {val_iou:.4f}")
    print(f"Early stopping: {full_history['early_stopped']}")
    print(f"================================================\n")
    
    # Generate training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(full_history['epochs'], full_history['train_loss'], label='Train Loss')
    plt.plot(full_history['epochs'], full_history['val_loss'], label='Validation Loss')
    plt.axvline(x=full_history['best_epoch'], color='r', linestyle='--', label=f'Best Epoch: {full_history["best_epoch"]}')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 2)
    plt.plot(full_history['epochs'], full_history['val_dice'], label='Dice')
    plt.plot(full_history['epochs'], full_history['val_iou'], label='IoU')
    plt.axvline(x=full_history['best_epoch'], color='r', linestyle='--', label=f'Best Epoch: {full_history["best_epoch"]}')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 3)
    plt.plot(full_history['epochs'], full_history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(unet_model_dir, 'unet_training_curves.png'))
    plt.close()
    
    # Only retain necessary chart code - these keys may not be present in history
    try:
        # Try plotting a bar chart of training time per epoch
        if 'time_per_epoch' in history and len(history['time_per_epoch']) > 0:
            plt.figure(figsize=(10, 5))
            plt.bar(full_history['epochs'], history['time_per_epoch'])
            plt.title('Training Time per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Time (seconds)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(unet_model_dir, 'unet_training_time.png'))
            plt.close()
        
        # Try plotting distribution on validation set
        if 'per_sample_dice' in history and len(history['per_sample_dice']) > 0:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            for epoch, dices in enumerate(history['per_sample_dice']):
                if epoch in [0, len(history['per_sample_dice'])//2, len(history['per_sample_dice'])-1]:
                    plt.hist(dices, bins=20, alpha=0.7, label=f'Epoch {epoch+1}')
            plt.title('Dice Score Distribution (Validation)')
            plt.xlabel('Dice Score')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.subplot(1, 2, 2)
            for epoch, ious in enumerate(history['per_sample_iou']):
                if epoch in [0, len(history['per_sample_iou'])//2, len(history['per_sample_iou'])-1]:
                    plt.hist(ious, bins=20, alpha=0.7, label=f'Epoch {epoch+1}')
            plt.title('IoU Score Distribution (Validation)')
            plt.xlabel('IoU Score')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(unet_model_dir, 'unet_score_distributions.png'))
            plt.close()
    except Exception as e:
        print(f"Error plotting additional charts: {str(e)}")
    
    # Load best model for evaluation
    print(f"Loading best U-Net model from epoch {full_history['best_epoch']} for evaluation...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded best U-Net model, validation Dice: {checkpoint['dice']:.4f}, IoU: {checkpoint['iou']:.4f}")
    
    return model, full_history

def train_rl_agent(args, device, data_loaders):
    """Train and evaluate the RL agent"""
    from src.train_rl import train_agent, evaluate_agent
    
    print('Training RL agent...')
    
    # Create feature extractor
    feature_extractor = PolypFeatureExtractor(device=device)
    
    # Set input dimension (extracted features + pointer location)
    input_dim = 4 * 16 * 16 + 2  # Features from a 4-channel 16x16 feature map + 2 for pointer location
    
    # Create PPO agent
    agent = PPOAgent(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=7,  # 7 actions
        feature_extractor=feature_extractor,
        device=device,
        lr=args.lr,
        gamma=0.99,
        clip_ratio=0.2,
        entropy_coef=0.01
    )
    
    # Train agent
    history = train_agent(
        agent=agent,
        train_dataset=data_loaders['train'].dataset,
        device=device,
        num_episodes=args.rl_episodes,
        update_interval=128,
        save_interval=100
    )
    
    # Return the trained agent
    return agent, history

def train_simple_rl_agent(args, device, data_loaders):
    """Train the SimpleRL agent from simple_rl.py"""
    from src.inter_rl import SimpleRLAgent, train_agent
    
    print('Training Simple RL agent...')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create agent
    agent = SimpleRLAgent(
        n_actions=7,  # 0=Up, 1=Down, 2=Left, 3=Right, 4=Expand, 5=Shrink, 6=Done
        lr=args.lr,
        gamma=0.99,
        device=device
    )
    
    # Use args directly
    train_agent(args)
    
    # Return agent
    return agent

def evaluate_unet(model, test_dataset, device, save_dir=None, num_samples=50, save_visualizations=False, model_info=None):
    """Evaluate the U-Net model on the test dataset and save results in JSON format"""
    from src.unet_model import dice_coefficient, iou_score
    import torch
    import matplotlib.pyplot as plt
    import os
    import json
    import time
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if save_visualizations:
            os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)
    
    model.eval()
    dice_scores = []
    iou_scores = []
    inference_times = []
    sample_details = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            # Get sample
            sample = test_dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            gt_mask = sample['mask'].unsqueeze(0).to(device)
            
            # Time the inference
            start_time = time.time()
            output = model(image)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            pred_mask = (output > 0.5).float()  # Use 0.5 threshold for sigmoid output
            
            # Calculate metrics
            dice = dice_coefficient(pred_mask, gt_mask).item()
            dice_scores.append(dice)
            
            iou = iou_score(pred_mask, gt_mask).item()
            iou_scores.append(iou)
            
            # Save sample details
            sample_details.append({
                'sample_idx': i,
                'dice': dice,
                'iou': iou,
                'inference_time': inference_time
            })
            
            # Save visualizations if requested
            if save_visualizations and save_dir and i < 5:  # Save only first 5 visualizations
                # Convert tensors to numpy for visualization
                image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
                gt_mask_np = gt_mask.squeeze().cpu().numpy()
                pred_mask_np = pred_mask.squeeze().cpu().numpy()
                
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.imshow(image_np)
                plt.title('Original Image')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(gt_mask_np, cmap='gray')
                plt.title('Ground Truth')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(pred_mask_np, cmap='gray')
                plt.title(f'Predicted Mask\nDice: {dice:.4f}, IoU: {iou:.4f}')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'visualizations', f'unet_sample_{i+1}.png'))
                plt.close()
    
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    mean_inference_time = np.mean(inference_times)
    std_dice = np.std(dice_scores)
    std_iou = np.std(iou_scores)
    
    # Create results dictionary
    results = {
        'model_type': 'unet',
        'mean_dice': float(mean_dice),
        'mean_iou': float(mean_iou),
        'mean_inference_time': float(mean_inference_time),
        'std_dice': float(std_dice),
        'std_iou': float(std_iou),
        'min_dice': float(np.min(dice_scores)),
        'max_dice': float(np.max(dice_scores)),
        'min_iou': float(np.min(iou_scores)),
        'max_iou': float(np.max(iou_scores)),
        'num_samples': len(dice_scores),
        'samples': sample_details,
        'evaluation_timestamp': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
    
    # Add model info if provided
    if model_info:
        results.update(model_info)
    
    # Save results to a JSON file if save_dir is provided
    if save_dir:
        with open(os.path.join(save_dir, 'unet_evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save a human-readable summary text file
        with open(os.path.join(save_dir, 'unet_evaluation_summary.txt'), 'w') as f:
            f.write(f'U-Net Evaluation Results\n')
            f.write(f'=======================\n\n')
            f.write(f'Mean Dice Score: {mean_dice:.4f} ± {std_dice:.4f}\n')
            f.write(f'Mean IoU Score: {mean_iou:.4f} ± {std_iou:.4f}\n')
            f.write(f'Mean Inference Time: {mean_inference_time*1000:.2f} ms per sample\n\n')
            f.write(f'Min/Max Dice: {np.min(dice_scores):.4f}/{np.max(dice_scores):.4f}\n')
            f.write(f'Min/Max IoU: {np.min(iou_scores):.4f}/{np.max(iou_scores):.4f}\n\n')
            f.write(f'Number of samples: {len(dice_scores)}\n')
    
    return results

def evaluate_simple_rl(agent, test_dataset, device, max_steps=20, save_dir=None, num_samples=50, save_visualizations=False, model_info=None):
    """Evaluate the SimpleRL agent on the test dataset and save results in JSON format"""
    from src.inter_rl import predict_mask, visualize_results
    from src.utils import dice_coefficient, iou_score
    import matplotlib.pyplot as plt
    import os
    import torch.nn.functional as F
    import json
    import time
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if save_visualizations:
            os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)
    
    dice_scores = []
    iou_scores = []
    inference_times = []
    steps_taken = []
    sample_details = []
    
    for i in range(min(num_samples, len(test_dataset))):
        try:
            # Get sample
            sample = test_dataset[i]
            image = sample['image'].numpy()
            gt_mask = sample['mask'].numpy()
            
            # Check and ensure the image has the expected shape
            expected_shape = (3, 288, 384)  # The shape expected by the SimplePolicyNetwork
            
            # Print the original shape for debugging
            if i == 0:  # Only print for the first sample
                print(f"Sample {i+1} image shape: {image.shape}")
            
            if image.shape != expected_shape:
                # Only print for the first sample that needs resizing
                if i == 0:
                    print(f"Resizing image from {image.shape} to {expected_shape}")
                    # Transpose to (H, W, C) for resizing if needed
                    if image.shape[0] == 3:
                        # Already in (C, H, W) format
                        image_for_resize = np.transpose(image, (1, 2, 0))
                        resized_image = cv2.resize(image_for_resize, (expected_shape[2], expected_shape[1]))
                        image = np.transpose(resized_image, (2, 0, 1))
                    else:
                        # Assumed to be in (H, W, C) format
                        resized_image = cv2.resize(image, (expected_shape[2], expected_shape[1]))
                        if resized_image.ndim == 2:
                            # If grayscale, add channel dimension
                            resized_image = np.expand_dims(resized_image, axis=0)
                        else:
                            # If color, transpose to (C, H, W)
                            image = np.transpose(resized_image, (2, 0, 1))
                
                # Also resize the mask
                if gt_mask.shape != (1, expected_shape[1], expected_shape[2]):
                    mask_for_resize = gt_mask.squeeze()
                    resized_mask = cv2.resize(mask_for_resize, (expected_shape[2], expected_shape[1]))
                    gt_mask = np.expand_dims(resized_mask, axis=0)
            
            # Predict mask and time it
            start_time = time.time()
            # Modify: Remove return_steps parameter, only get mask
            pred_mask = predict_mask(agent, image, gt_mask, max_steps=max_steps)
            inference_time = time.time() - start_time
            
            # Set max_steps as the number of steps, as we cannot know the actual number of steps
            num_steps = max_steps  # Note: This will cause all samples to use the maximum number of steps
            
            inference_times.append(inference_time)
            steps_taken.append(num_steps)
            
            # Calculate metrics
            dice = dice_coefficient(torch.tensor(pred_mask), torch.tensor(gt_mask))
            dice_scores.append(dice.item())
            
            iou = iou_score(torch.tensor(pred_mask), torch.tensor(gt_mask))
            iou_scores.append(iou.item())
            
            # Save sample details
            sample_details.append({
                'sample_idx': i,
                'dice': dice.item(),
                'iou': iou.item(),
                'inference_time': inference_time,
                'steps_taken': num_steps
            })
            
            # Visualize samples if requested
            if save_visualizations and save_dir and i < 5:  # Only visualize the first 5 samples
                vis_img = visualize_results(image, gt_mask, pred_mask)
                plt.figure(figsize=(8, 8))
                plt.imshow(vis_img)
                plt.title(f'SimpleRL Segmentation\nDice: {dice.item():.4f}, IoU: {iou.item():.4f}\nSteps: {num_steps}/{max_steps}')
                plt.axis('off')
                plt.savefig(os.path.join(save_dir, 'visualizations', f'simple_rl_sample_{i+1}.png'), bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Error processing sample {i+1}: {str(e)}")
            continue
    
    if not dice_scores:
        print("No valid predictions were made. Check if the model is compatible with the input data.")
        return {
            'model_type': 'simple_rl',
            'mean_dice': 0.0,
            'mean_iou': 0.0,
            'mean_inference_time': 0.0,
            'mean_steps': 0.0,
            'samples': [],
            'error': "No valid predictions were made"
        }
    
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    mean_inference_time = np.mean(inference_times)
    mean_steps = np.mean(steps_taken)
    std_dice = np.std(dice_scores)
    std_iou = np.std(iou_scores)
    std_steps = np.std(steps_taken)
    
    # Create results dictionary
    results = {
        'model_type': 'simple_rl',
        'mean_dice': float(mean_dice),
        'mean_iou': float(mean_iou),
        'mean_inference_time': float(mean_inference_time),
        'mean_steps': float(mean_steps),
        'std_dice': float(std_dice),
        'std_iou': float(std_iou),
        'std_steps': float(std_steps),
        'min_dice': float(np.min(dice_scores)),
        'max_dice': float(np.max(dice_scores)),
        'min_iou': float(np.min(iou_scores)),
        'max_iou': float(np.max(iou_scores)),
        'min_steps': int(np.min(steps_taken)),
        'max_steps': int(np.max(steps_taken)),
        'num_samples': len(dice_scores),
        'samples': sample_details,
        'max_allowed_steps': max_steps,
        'evaluation_timestamp': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
    
    # Add model info if provided
    if model_info:
        results.update(model_info)
    
    # Save results to a JSON file if save_dir is provided
    if save_dir:
        with open(os.path.join(save_dir, 'simple_rl_evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save a human-readable summary text file
        with open(os.path.join(save_dir, 'simple_rl_evaluation_summary.txt'), 'w') as f:
            f.write(f'SimpleRL Evaluation Results\n')
            f.write(f'==========================\n\n')
            f.write(f'Mean Dice Score: {mean_dice:.4f} ± {std_dice:.4f}\n')
            f.write(f'Mean IoU Score: {mean_iou:.4f} ± {std_iou:.4f}\n')
            f.write(f'Mean Inference Time: {mean_inference_time*1000:.2f} ms per sample\n')
            f.write(f'Mean Steps Taken: {mean_steps:.2f} ± {std_steps:.2f} out of {max_steps} max steps\n\n')
            f.write(f'Min/Max Dice: {np.min(dice_scores):.4f}/{np.max(dice_scores):.4f}\n')
            f.write(f'Min/Max IoU: {np.min(iou_scores):.4f}/{np.max(iou_scores):.4f}\n')
            f.write(f'Min/Max Steps: {np.min(steps_taken)}/{np.max(steps_taken)}\n\n')
            f.write(f'Number of samples: {len(dice_scores)}\n')
    
    return results

def compare_unet_simple_rl(args, device, data_loaders):
    """Compare U-Net and SimpleRL on the same dataset"""
    print('Comparing U-Net and SimpleRL...')
    
    results = {}
    
    # Train U-Net for specified epochs
    unet_model, unet_history = train_unet_model(args, device, data_loaders)
    
    # Evaluate U-Net
    from src.unet_model import dice_coefficient, iou_score
    
    unet_dice = 0
    unet_iou = 0
    
    with torch.no_grad():
        for batch in data_loaders['test']:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = unet_model(images)
            pred_masks = (outputs > 0.5).float()
            
            unet_dice += dice_coefficient(pred_masks, masks).item()
            unet_iou += iou_score(pred_masks, masks).item()
    
    unet_dice /= len(data_loaders['test'])
    unet_iou /= len(data_loaders['test'])
    
    results['unet'] = {
        'dice': unet_dice,
        'iou': unet_iou
    }
    
    print(f'U-Net Results:')
    print(f'  - Dice: {unet_dice:.4f}')
    print(f'  - IoU: {unet_iou:.4f}')
    
    # Train SimpleRL
    simple_rl_agent = train_simple_rl_agent(args, device, data_loaders)
    
    # Load the best model
    from src.inter_rl import SimpleRLAgent
    best_agent = SimpleRLAgent(
        n_actions=7,
        lr=args.lr,
        gamma=0.99,
        device=device
    )
    
    try:
        best_agent.load('results/simple_rl/best_model.pth')
        
        # Evaluate SimpleRL
        simple_rl_results = evaluate_simple_rl(
            agent=best_agent,
            test_dataset=data_loaders['test'].dataset,
            device=device,
            max_steps=args.max_steps
        )
        
        results['simple_rl'] = {
            'dice': simple_rl_results['mean_dice'],
            'iou': simple_rl_results['mean_iou']
        }
        
        print(f'SimpleRL Results:')
        print(f'  - Dice: {results["simple_rl"]["dice"]:.4f}')
        print(f'  - IoU: {results["simple_rl"]["iou"]:.4f}')
        
        # Compare results
        print('\nModel Comparison:')
        print(f'                U-Net      SimpleRL')
        print(f'Dice:           {results["unet"]["dice"]:.4f}      {results["simple_rl"]["dice"]:.4f}')
        print(f'IoU:            {results["unet"]["iou"]:.4f}      {results["simple_rl"]["iou"]:.4f}')
        
    except:
        print('Could not load best SimpleRL model. Skipping comparison.')
    
    return results

def evaluate_models(args, device, data_loaders, unet_model=None, rl_agent=None):
    """Evaluate and compare the U-Net model and RL agent"""
    from src.train_rl import evaluate_agent
    
    print('Evaluating models...')
    
    results = {}
    
    # Load models if not provided
    if unet_model is None:
        unet_model = UNet(n_channels=3, n_classes=1, bilinear=True)
        try:
            unet_model.load_state_dict(torch.load('models/unet_model.pth', map_location=device))
            unet_model = unet_model.to(device)
            unet_model.eval()
        except:
            print('Could not load U-Net model. Skipping U-Net evaluation.')
            unet_model = None
    
    if rl_agent is None:
        feature_extractor = PolypFeatureExtractor(device=device)
        input_dim = 4 * 16 * 16 + 2
        
        rl_agent = PPOAgent(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=7,
            feature_extractor=feature_extractor,
            device=device
        )
        
        try:
            rl_agent.load('models/ppo_agent_final.pth')
        except:
            print('Could not load RL agent. Skipping RL evaluation.')
            rl_agent = None
    
    # Evaluate U-Net model
    if unet_model is not None:
        from src.unet_model import dice_coefficient, iou_score
        
        unet_dice = 0
        unet_iou = 0
        
        with torch.no_grad():
            for batch in data_loaders['test']:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = unet_model(images)
                pred_masks = (outputs > 0.5).float()
                
                unet_dice += dice_coefficient(pred_masks, masks).item()
                unet_iou += iou_score(pred_masks, masks).item()
        
        unet_dice /= len(data_loaders['test'])
        unet_iou /= len(data_loaders['test'])
        
        results['unet'] = {
            'dice': unet_dice,
            'iou': unet_iou
        }
        
        print(f'U-Net Results:')
        print(f'  - Dice: {unet_dice:.4f}')
        print(f'  - IoU: {unet_iou:.4f}')
    
    # Evaluate RL agent
    if rl_agent is not None:
        rl_results = evaluate_agent(
            agent=rl_agent,
            test_dataset=data_loaders['test'].dataset,
            device=device,
            num_episodes=100
        )
        
        results['rl'] = {
            'dice': np.mean(rl_results['dice_scores']),
            'iou': np.mean(rl_results['iou_scores']),
            'steps': np.mean(rl_results['steps'])
        }
        
        print(f'RL Agent Results:')
        print(f'  - Dice: {results["rl"]["dice"]:.4f}')
        print(f'  - IoU: {results["rl"]["iou"]:.4f}')
        print(f'  - Average Steps: {results["rl"]["steps"]:.2f}')
    
    # Compare models
    if 'unet' in results and 'rl' in results:
        print('\nModel Comparison:')
        print(f'                U-Net      RL Agent')
        print(f'Dice:           {results["unet"]["dice"]:.4f}      {results["rl"]["dice"]:.4f}')
        print(f'IoU:            {results["unet"]["iou"]:.4f}      {results["rl"]["iou"]:.4f}')
        
        # Additional RL metrics
        print(f'\nAdditional RL Metrics:')
        print(f'Average Steps: {results["rl"]["steps"]:.2f}')
    
    return results

def train_and_evaluate(args, device, data_loaders):
    """Train both U-Net and SimpleRL models, save them, and then evaluate and compare them"""
    # Import necessary functions
    from src.unet_model import dice_coefficient, iou_score, DiceLoss
    from src.simple_rl import train_agent
    import numpy as np
    import pickle
    
    print('Training and evaluating models...')
    
    # Create results directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results/comparison_{timestamp}'
    models_dir = f'models/checkpoints_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, 'unet'), exist_ok=True)
    os.makedirs(os.path.join(models_dir, 'simple_rl'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'unet_data'), exist_ok=True)
    
    # Save training configuration
    with open(os.path.join(results_dir, 'training_config.txt'), 'w') as f:
        f.write(f'Training Configuration\n')
        f.write(f'======================\n\n')
        f.write(f'U-Net Epochs: {args.unet_epochs}\n')
        f.write(f'SimpleRL Episodes: {args.simple_rl_episodes}\n')
        f.write(f'Max Steps per Episode: {args.max_steps}\n')
        f.write(f'Batch Size: {args.batch_size}\n')
        f.write(f'Learning Rate: {args.lr}\n')
        f.write(f'Device: {device}\n')
        f.write(f'Timestamp: {timestamp}\n')
    
    # 1. Train U-Net model and save checkpoint
    print('\n1. Training U-Net model...')
    
    # Create U-Net model
    unet_model = UNet(n_channels=3, n_classes=1, bilinear=True)
    unet_model = unet_model.to(device)
    
    # Create U-Net optimizer
    optimizer = torch.optim.AdamW(unet_model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Use combined loss function: BCE + Dice Loss
    bce_criterion = nn.BCELoss()
    dice_criterion = DiceLoss()
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=True
    )
    
    # For early stopping and saving best model
    best_val_dice = 0.0
    best_epoch = 0
    patience = 15  # Early stopping patience value
    patience_counter = 0
    
    # Training history record
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'lr': [],
        'epochs': [],
        'time_per_epoch': [],
        'batch_losses': [],  # Record losses for each batch
        'per_sample_dice': [],  # Dice score for each sample
        'per_sample_iou': []    # IoU score for each sample
    }
    
    for epoch in range(args.unet_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        unet_model.train()
        train_loss = 0
        batch_losses = []
        
        for batch_idx, batch in enumerate(data_loaders['train']):
            # Get images and masks
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = unet_model(images)
            
            # Calculate loss - combined BCE and Dice Loss
            bce_loss = bce_criterion(outputs, masks)
            dice_loss = dice_criterion(outputs, masks)
            loss = bce_loss + dice_loss
            
            # Record loss for each batch
            batch_losses.append(loss.item())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet_model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.unet_epochs} | Batch {batch_idx}/{len(data_loaders['train'])} | Loss: {loss.item():.4f}")
        
        train_loss /= len(data_loaders['train'])
        history['train_loss'].append(train_loss)
        history['batch_losses'].append(batch_losses)
        
        # Validation phase
        unet_model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        epoch_per_sample_dice = []
        epoch_per_sample_iou = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loaders['val']):
                # Get images and masks
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Forward pass
                outputs = unet_model(images)
                
                # Calculate loss
                bce_loss = bce_criterion(outputs, masks)
                dice_loss = dice_criterion(outputs, masks)
                loss = bce_loss + dice_loss
                val_loss += loss.item()
                
                # Generate binary mask predictions, note threshold is 0.5
                pred_masks = (outputs > 0.5).float()
                
                # Calculate metrics per sample
                for i in range(pred_masks.size(0)):
                    sample_dice = dice_coefficient(pred_masks[i:i+1], masks[i:i+1]).item()
                    sample_iou = iou_score(pred_masks[i:i+1], masks[i:i+1]).item()
                    
                    epoch_per_sample_dice.append(sample_dice)
                    epoch_per_sample_iou.append(sample_iou)
                
                # Calculate batch-level metrics
                batch_dice = dice_coefficient(pred_masks, masks).item()
                batch_iou = iou_score(pred_masks, masks).item()
                
                val_dice += batch_dice
                val_iou += batch_iou
        
        val_loss /= len(data_loaders['val'])
        val_dice /= len(data_loaders['val'])
        val_iou /= len(data_loaders['val'])
        
        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Record training time for each epoch
        epoch_time = time.time() - epoch_start_time
        
        # Record validation metrics
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['lr'].append(current_lr)
        history['epochs'].append(epoch + 1)
        history['time_per_epoch'].append(epoch_time)
        history['per_sample_dice'].append(epoch_per_sample_dice)
        history['per_sample_iou'].append(epoch_per_sample_iou)
        
        print(f"Epoch {epoch+1}/{args.unet_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Time: {epoch_time:.2f}s")
        
        # Save training data for each epoch
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'lr': current_lr,
            'time': epoch_time,
            'batch_losses': batch_losses,
            'per_sample_dice': epoch_per_sample_dice,
            'per_sample_iou': epoch_per_sample_iou
        }
        
        # Save detailed data for each epoch
        with open(os.path.join(results_dir, 'unet_data', f'epoch_{epoch+1}_data.pkl'), 'wb') as f:
            pickle.dump(epoch_data, f)
        
        # Visualize validation results for the current epoch
        if args.save_visualizations:
            # Randomly select a few validation samples for visualization
            with torch.no_grad():
                sample_indices = np.random.randint(0, len(data_loaders['val'].dataset), 3)
                
                for i, idx in enumerate(sample_indices):
                    sample = data_loaders['val'].dataset[idx]
                    image = sample['image'].unsqueeze(0).to(device)
                    gt_mask = sample['mask'].unsqueeze(0).to(device)
                    
                    output = unet_model(image)
                    pred_mask = (output > 0.5).float()
                    
                    # Calculate metrics
                    sample_dice = dice_coefficient(pred_mask, gt_mask).item()
                    sample_iou = iou_score(pred_mask, gt_mask).item()
                    
                    # Convert to numpy
                    image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
                    gt_mask_np = gt_mask.squeeze().cpu().numpy()
                    pred_mask_np = pred_mask.squeeze().cpu().numpy()
                    
                    plt.figure(figsize=(12, 4))
                    
                    plt.subplot(1, 3, 1)
                    plt.imshow(image_np)
                    plt.title('Original Image')
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 2)
                    plt.imshow(gt_mask_np, cmap='gray')
                    plt.title('Ground Truth')
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 3)
                    plt.imshow(pred_mask_np, cmap='gray')
                    plt.title(f'Predicted Mask\nDice: {sample_dice:.4f}, IoU: {sample_iou:.4f}')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    os.makedirs(os.path.join(results_dir, 'unet_data', f'epoch_{epoch+1}_viz'), exist_ok=True)
                    plt.savefig(os.path.join(results_dir, 'unet_data', f'epoch_{epoch+1}_viz', f'sample_{i+1}.png'))
                    plt.close()
        
        # Save best model, use try-except block to handle potential disk space issues
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_path = os.path.join(models_dir, 'unet', 'unet_best_model.pth')
            
            # Use try-except block to save best model
            try:
                # Save best model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': unet_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': val_loss,
                    'dice': val_dice,
                    'iou': val_iou
                }, best_model_path)
                
                print(f"Saved new best U-Net model, validation Dice score: {best_val_dice:.4f}")
            except Exception as e:
                print(f"Error saving best model: {str(e)}, skipping save and continuing training")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
            break
    
    # Save U-Net model and training history
    print(f"U-Net training completed. Best validation Dice score: {best_val_dice:.4f} at epoch {best_epoch}")
    
    # Save final model
    final_model_path = os.path.join(models_dir, 'unet', 'unet_final_model.pth')
    torch.save({
        'epoch': args.unet_epochs,
        'model_state_dict': unet_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_dice': val_dice,
        'val_iou': val_iou
    }, final_model_path)
    
    # Save training history
    with open(os.path.join(results_dir, 'unet_training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # Visualize training statistics
    plt.figure(figsize=(20, 10))
    
    # Plot loss
    plt.subplot(2, 3, 1)
    plt.plot(history['epochs'], history['train_loss'], label='Train Loss')
    plt.plot(history['epochs'], history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('U-Net Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Dice coefficient
    plt.subplot(2, 3, 2)
    plt.plot(history['epochs'], history['val_dice'], label='Validation Dice', color='g')
    plt.axhline(y=best_val_dice, color='g', linestyle='--', label=f'Best Dice: {best_val_dice:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('U-Net Dice Coefficient')
    plt.legend()
    plt.grid(True)
    
    # Plot IoU
    plt.subplot(2, 3, 3)
    plt.plot(history['epochs'], history['val_iou'], label='Validation IoU', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.title('U-Net IoU Score')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 3, 4)
    plt.semilogy(history['epochs'], history['lr'], label='Learning Rate', color='c')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate (log scale)')
    plt.title('U-Net Learning Rate')
    plt.legend()
    plt.grid(True)
    
    # Plot average time per epoch
    plt.subplot(2, 3, 5)
    plt.plot(history['epochs'], history['time_per_epoch'], label='Time per Epoch', color='m')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('U-Net Time per Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'unet_training_stats.png'))
    plt.close()
    
    # 2. 训练SimpleRL模型
    print('\n2. Training SimpleRL model...')
    
    # Create a new SimpleRL agent
    from src.inter_rl import SimpleRLAgent
    
    # Start training
    agent, simple_rl_history = train_agent(
        data_loaders['train'],
        data_loaders['val'],
        device=device,
        epochs=args.simple_rl_episodes,
        max_steps=args.max_steps,
        save_dir=os.path.join(models_dir, 'simple_rl'),
        results_dir=os.path.join(results_dir, 'simple_rl_data'),
        lr=args.lr,
        batch_size=args.batch_size,
        save_interval=max(1, args.simple_rl_episodes // 10),
        visualize=args.save_visualizations
    )
    
    # Save final SimpleRL model
    agent.save(os.path.join(models_dir, 'simple_rl', 'simple_rl_final.pth'))
    
    # Save model training history
    with open(os.path.join(results_dir, 'simple_rl_training_history.pkl'), 'wb') as f:
        pickle.dump(simple_rl_history, f)
    
    # 3. Compare U-Net vs SimpleRL on test set
    print('\n3. Comparing U-Net vs SimpleRL on test dataset...')
    
    # Evaluate both models on test set
    from src.unet_model import evaluate_unet
    from src.inter_rl import evaluate_simple_rl
    
    # Test dataset for evaluation
    test_dataset = data_loaders['test'].dataset
    
    # Initialize evaluation results dictionary
    eval_results = {
        'unet': {
            'dice': [],
            'iou': [],
            'inference_time': [],
            'sample_indices': []
        },
        'simple_rl': {
            'dice': [],
            'iou': [],
            'inference_time': [],
            'steps_taken': [],
            'sample_indices': []
        }
    }
    
    # 4. Evaluate models
    print('\n4. Evaluating models on test set...')
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Randomly select test samples for evaluation
    num_samples = min(args.num_eval_samples, len(test_dataset))
    sample_indices = np.random.choice(range(len(test_dataset)), num_samples, replace=False)
    
    # Create results visualization directory
    viz_dir = os.path.join(results_dir, 'comparison_viz')
    os.makedirs(viz_dir, exist_ok=True)
    
    # If U-Net model is available
    if unet_model is not None:
        print("Evaluating U-Net model...")
        # Evaluate on test set
        unet_results = evaluate_unet(
            unet_model, 
            test_dataset, 
            sample_indices, 
            device,
            viz_dir=viz_dir if args.save_visualizations else None
        )
        # Add results to evaluation dictionary
        eval_results['unet'] = unet_results
    
    # If SimpleRL model is available
    if agent is not None:
        print("Evaluating SimpleRL model...")
        # Evaluate on test set
        simple_rl_results = evaluate_simple_rl(
            agent, 
            test_dataset, 
            sample_indices, 
            device,
            max_steps=args.max_steps,
            viz_dir=viz_dir if args.save_visualizations else None
        )
        # Add results to evaluation dictionary
        eval_results['simple_rl'] = simple_rl_results
        
        # Add training validation results to evaluation results
        if isinstance(simple_rl_history, dict) and 'best_val_dice' in simple_rl_history:
            simple_rl_results['best_val_dice'] = simple_rl_history['best_val_dice']
            simple_rl_results['best_val_iou'] = simple_rl_history.get('best_val_iou', 0)
            simple_rl_results['best_episode'] = simple_rl_history.get('best_episode', 0)
    
    # 5. Compare results
    print('\n5. Comparing model results...')
    
    # Create comparison report
    with open(os.path.join(results_dir, 'comparison_report.txt'), 'w') as f:
        f.write('Model Comparison Report\n')
        f.write('=====================\n\n')
        
        # U-Net results
        f.write('U-Net Results\n')
        f.write('-----------\n')
        if 'mean_dice' in eval_results['unet']:
            f.write(f"Dice Coefficient: {eval_results['unet']['mean_dice']:.4f} (±{eval_results['unet']['std_dice']:.4f})\n")
            f.write(f"IoU Score: {eval_results['unet']['mean_iou']:.4f} (±{eval_results['unet']['std_iou']:.4f})\n")
            f.write(f"Average Inference Time: {eval_results['unet']['mean_inference_time']:.4f} seconds\n")
            f.write(f"Best Validation Dice: {best_val_dice:.4f} (Epoch {best_epoch})\n")
        else:
            f.write('No U-Net evaluation results available\n')
        
        f.write('\n')
        
        # SimpleRL results
        f.write('SimpleRL Results\n')
        f.write('---------------\n')
        if 'mean_dice' in eval_results['simple_rl']:
            f.write(f"Dice Coefficient: {eval_results['simple_rl']['mean_dice']:.4f} (±{eval_results['simple_rl']['std_dice']:.4f})\n")
            f.write(f"IoU Score: {eval_results['simple_rl']['mean_iou']:.4f} (±{eval_results['simple_rl']['std_iou']:.4f})\n")
            f.write(f"Average Inference Time: {eval_results['simple_rl']['mean_inference_time']:.4f} seconds\n")
            f.write(f"Average Steps Taken: {eval_results['simple_rl']['mean_steps_taken']:.2f} (±{eval_results['simple_rl']['std_steps_taken']:.2f})\n")
            if 'best_val_dice' in eval_results['simple_rl']:
                f.write(f"Best Validation Dice: {eval_results['simple_rl']['best_val_dice']:.4f} (Episode {eval_results['simple_rl']['best_episode']})\n")
        else:
            f.write('No SimpleRL evaluation results available\n')
        
        f.write('\n')
        
        # Comparison
        f.write('Comparison\n')
        f.write('----------\n')
        if 'mean_dice' in eval_results['unet'] and 'mean_dice' in eval_results['simple_rl']:
            # Dice comparison
            dice_diff = eval_results['simple_rl']['mean_dice'] - eval_results['unet']['mean_dice']
            dice_relative = dice_diff / eval_results['unet']['mean_dice'] * 100
            f.write(f"Dice Difference: {dice_diff:.4f} ({dice_relative:+.2f}%)\n")
            
            # IoU comparison
            iou_diff = eval_results['simple_rl']['mean_iou'] - eval_results['unet']['mean_iou']
            iou_relative = iou_diff / eval_results['unet']['mean_iou'] * 100
            f.write(f"IoU Difference: {iou_diff:.4f} ({iou_relative:+.2f}%)\n")
            
            # Time comparison
            time_diff = eval_results['simple_rl']['mean_inference_time'] - eval_results['unet']['mean_inference_time']
            time_relative = time_diff / eval_results['unet']['mean_inference_time'] * 100
            f.write(f"Inference Time Difference: {time_diff:.4f} seconds ({time_relative:+.2f}%)\n")
            
            # Overall assessment
            f.write('\nOverall Assessment:\n')
            if dice_diff > 0:
                f.write(f"SimpleRL outperforms U-Net in segmentation quality by {dice_relative:.2f}%\n")
            else:
                f.write(f"U-Net outperforms SimpleRL in segmentation quality by {-dice_relative:.2f}%\n")
                
            if time_diff < 0:
                f.write(f"SimpleRL is faster than U-Net by {-time_relative:.2f}%\n")
            else:
                f.write(f"U-Net is faster than SimpleRL by {time_relative:.2f}%\n")
        else:
            f.write('Cannot compare models: incomplete evaluation results\n')
    
    # Save all evaluation results
    with open(os.path.join(results_dir, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump(eval_results, f)
    
    # Create visualization if requested
    if args.save_visualizations and 'mean_dice' in eval_results['unet'] and 'mean_dice' in eval_results['simple_rl']:
        print("Creating comparison visualizations...")
        
        # Plot Dice and IoU comparison
        plt.figure(figsize=(12, 5))
        
        # Plot Dice comparison
        plt.subplot(1, 2, 1)
        models = ['U-Net', 'SimpleRL']
        dice_values = [eval_results['unet']['mean_dice'], eval_results['simple_rl']['mean_dice']]
        dice_error = [eval_results['unet']['std_dice'], eval_results['simple_rl']['std_dice']]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x, dice_values, width, yerr=dice_error, capsize=10, label='Dice Coefficient')
        plt.ylabel('Dice Coefficient')
        plt.title('Dice Coefficient Comparison')
        plt.xticks(x, models)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot IoU comparison
        plt.subplot(1, 2, 2)
        iou_values = [eval_results['unet']['mean_iou'], eval_results['simple_rl']['mean_iou']]
        iou_error = [eval_results['unet']['std_iou'], eval_results['simple_rl']['std_iou']]
        
        plt.bar(x, iou_values, width, yerr=iou_error, capsize=10, label='IoU Score')
        plt.ylabel('IoU Score')
        plt.title('IoU Score Comparison')
        plt.xticks(x, models)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'metrics_comparison.png'))
        plt.close()
        
        # Plot inference time and steps for SimpleRL
        plt.figure(figsize=(12, 5))
        
        # Plot inference time comparison
        plt.subplot(1, 2, 1)
        time_values = [eval_results['unet']['mean_inference_time'], eval_results['simple_rl']['mean_inference_time']]
        time_error = [eval_results['unet']['std_inference_time'], eval_results['simple_rl']['std_inference_time']]
        
        plt.bar(x, time_values, width, yerr=time_error, capsize=10, label='Inference Time')
        plt.ylabel('Time (seconds)')
        plt.title('Inference Time Comparison')
        plt.xticks(x, models)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot steps for SimpleRL
        plt.subplot(1, 2, 2)
        plt.hist(eval_results['simple_rl']['steps_taken'], bins=20, alpha=0.7)
        plt.axvline(
            x=eval_results['simple_rl']['mean_steps_taken'], 
            color='r', 
            linestyle='--', 
            label=f"Mean: {eval_results['simple_rl']['mean_steps_taken']:.2f}"
        )
        plt.xlabel('Steps')
        plt.ylabel('Frequency')
        plt.title('SimpleRL Steps Distribution')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'performance_comparison.png'))
        plt.close()
    
    print(f"\nTraining and evaluation completed. Results saved in {results_dir}")
    
    # Return evaluation results for further analysis if needed
    return eval_results

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set up environment with specified seed
    device = setup_environment(seed=args.seed)
    print(f'Using device: {device}')
    
    # Parse split ratio
    try:
        split_ratio = tuple(float(x) for x in args.split_ratio.split(','))
        assert len(split_ratio) == 3, "Split ratio must have 3 values"
        assert sum(split_ratio) == 1.0, "Split ratio must sum to 1.0"
    except:
        print(f"Invalid split ratio: {args.split_ratio}, using default (0.7, 0.15, 0.15)")
        split_ratio = (0.7, 0.15, 0.15)
    
    # Get improved data loaders with better shuffling
    data_loaders = get_improved_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2,
        seed=args.seed,
        split_ratio=split_ratio
    )
    
    # Create timestamped output directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results/run_{timestamp}'
    models_dir = f'models/run_{timestamp}'
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the run configuration
    run_config = {
        'timestamp': timestamp,
        'mode': args.mode,
        'seed': args.seed,
        'data_dir': args.data_dir,
        'split_ratio': list(split_ratio),
        'unet_epochs': args.unet_epochs,
        'simple_rl_episodes': args.simple_rl_episodes,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'max_steps': args.max_steps,
        'device': str(device),
        'dataset_size': {
            'train': len(data_loaders['train_dataset']),
            'val': len(data_loaders['val_dataset']),
            'test': len(data_loaders['test_dataset']),
            'total': len(data_loaders['full_dataset'])
        }
    }
    
    with open(os.path.join(results_dir, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    
    # Based on the mode, run the appropriate functions
    unet_model = None
    rl_agent = None
    unet_result = None
    simple_rl_result = None
    
    # Make directories for each model
    unet_dir = os.path.join(results_dir, 'unet')
    simple_rl_dir = os.path.join(results_dir, 'simple_rl')
    os.makedirs(unet_dir, exist_ok=True)
    os.makedirs(simple_rl_dir, exist_ok=True)
    
    if args.mode == 'train_unet' or args.mode == 'train_and_evaluate' or args.mode == 'all' or args.mode == 'compare_unet_simple_rl':
        # Train U-Net model
        print("\n===== Training U-Net Model =====")
        unet_model, unet_history = train_unet_model(args, device, data_loaders)
    
        # Save U-Net history
        with open(os.path.join(unet_dir, 'training_history.json'), 'w') as f:
            # Convert any numpy values to Python native types for JSON serialization
            history_copy = {}
            for key, value in unet_history.items():
                if isinstance(value, list):
                    # If the values in the list are numpy types, convert them
                    if value and isinstance(value[0], (np.integer, np.floating)):
                        history_copy[key] = [float(v) for v in value]
                    elif value and isinstance(value[0], list):
                        # For nested lists (like per_sample_dice)
                        history_copy[key] = [[float(v) for v in sublist] for sublist in value]
                    else:
                        history_copy[key] = value
                elif isinstance(value, (np.integer, np.floating)):
                    history_copy[key] = float(value)
                else:
                    history_copy[key] = value
            
            json.dump(history_copy, f, indent=2)
        
        # Evaluate U-Net on test set
        unet_model.eval()
        model_info = {
            'training_epochs': args.unet_epochs,
            'learning_rate': args.lr,
            'batch_size': args.batch_size
        }
        
        unet_result = evaluate_unet(
            model=unet_model,
            test_dataset=data_loaders['test_dataset'],
            device=device,
            save_dir=unet_dir,
            num_samples=args.eval_samples,
            save_visualizations=args.save_visualizations,
            model_info=model_info
        )
        
        print(f"U-Net Evaluation Results:")
        print(f"  - Dice: {unet_result['mean_dice']:.4f} ± {unet_result['std_dice']:.4f}")
        print(f"  - IoU: {unet_result['mean_iou']:.4f} ± {unet_result['std_iou']:.4f}")
    
    if args.mode == 'train_simple_rl' or args.mode == 'train_and_evaluate' or args.mode == 'compare_unet_simple_rl':
        # Train SimpleRL agent
        print("\n===== Training SimpleRL Agent =====")
        
        # Create args for SimpleRL training
        simple_rl_args = argparse.Namespace(
            data_dir=args.data_dir,
            output_dir=simple_rl_dir,
            lr=args.lr,
            gamma=0.99,
            num_episodes=args.simple_rl_episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            update_interval=5,
            log_interval=5,
            eval_interval=args.eval_interval,
            num_eval_episodes=args.num_eval_episodes,
            save_interval=10,
            device=device,
            seed=args.seed
        )
        
        # Train SimpleRL agent
        from src.inter_rl import train_agent, SimpleRLAgent
        
        training_results = train_agent(simple_rl_args)
        
        # Load the best model for evaluation
        simple_rl_agent = SimpleRLAgent(
            n_actions=7,
            lr=args.lr,
            gamma=0.99,
            device=device
        )
        
        try:
            best_model_path = os.path.join(simple_rl_dir, 'best_model.pth')
            simple_rl_agent.load(best_model_path)
            
            # Evaluate SimpleRL agent
            model_info = {
                'training_episodes': args.simple_rl_episodes,
                'max_steps': args.max_steps,
                'learning_rate': args.lr,
                'gamma': 0.99,
                'best_val_dice': float(training_results.get('best_val_dice', 0)),
                'best_val_iou': float(training_results.get('best_val_iou', 0)),
                'best_episode': int(training_results.get('best_episode', 0))
            }
            
            simple_rl_result = evaluate_simple_rl(
                agent=simple_rl_agent,
                test_dataset=data_loaders['test_dataset'],
                device=device,
                max_steps=args.max_steps,
                save_dir=simple_rl_dir,
                num_samples=args.eval_samples,
                save_visualizations=args.save_visualizations,
                model_info=model_info
            )
            
            print(f"SimpleRL Evaluation Results:")
            print(f"  - Dice: {simple_rl_result['mean_dice']:.4f} ± {simple_rl_result['std_dice']:.4f}")
            print(f"  - IoU: {simple_rl_result['mean_iou']:.4f} ± {simple_rl_result['std_iou']:.4f}")
            print(f"  - Avg Steps: {simple_rl_result['mean_steps']:.2f} ± {simple_rl_result['std_steps']:.2f}")
        except Exception as e:
            print(f"Error loading or evaluating SimpleRL model: {str(e)}")
    
    # For evaluation mode, try to load existing models
    if args.mode == 'evaluate':
        print("\n===== Evaluating Existing Models =====")
        
        # Try to load U-Net model
        try:
            unet_model = UNet(n_channels=3, n_classes=1, bilinear=True)
            unet_model.load_state_dict(torch.load('models/unet_model.pth', map_location=device))
            unet_model = unet_model.to(device)
            
            unet_result = evaluate_unet(
                model=unet_model,
                test_dataset=data_loaders['test_dataset'],
                device=device,
                save_dir=unet_dir,
                num_samples=args.eval_samples,
                save_visualizations=args.save_visualizations
            )
            
            print(f"U-Net Evaluation Results:")
            print(f"  - Dice: {unet_result['mean_dice']:.4f} ± {unet_result['std_dice']:.4f}")
            print(f"  - IoU: {unet_result['mean_iou']:.4f} ± {unet_result['std_iou']:.4f}")
        except Exception as e:
            print(f"Error loading or evaluating U-Net model: {str(e)}")
        
        # Try to load SimpleRL model
        try:
            from src.inter_rl import SimpleRLAgent
            
            simple_rl_agent = SimpleRLAgent(
                n_actions=7,
                lr=args.lr,
                gamma=0.99,
                device=device
            )
            
            simple_rl_agent.load('models/simple_rl/best_model.pth')
            
            simple_rl_result = evaluate_simple_rl(
                agent=simple_rl_agent,
                test_dataset=data_loaders['test_dataset'],
                device=device,
                max_steps=args.max_steps,
                save_dir=simple_rl_dir,
                num_samples=args.eval_samples,
                save_visualizations=args.save_visualizations
            )
            
            print(f"SimpleRL Evaluation Results:")
            print(f"  - Dice: {simple_rl_result['mean_dice']:.4f} ± {simple_rl_result['std_dice']:.4f}")
            print(f"  - IoU: {simple_rl_result['mean_iou']:.4f} ± {simple_rl_result['std_iou']:.4f}")
            print(f"  - Avg Steps: {simple_rl_result['mean_steps']:.2f} ± {simple_rl_result['std_steps']:.2f}")
        except Exception as e:
            print(f"Error loading or evaluating SimpleRL model: {str(e)}")
    
    # Compare model results if both are available
    if unet_result and simple_rl_result:
        print("\n===== Model Comparison =====")
        
        comparison = {
            'timestamp': timestamp,
            'unet': {
                'mean_dice': unet_result['mean_dice'],
                'mean_iou': unet_result['mean_iou'],
                'std_dice': unet_result['std_dice'],
                'std_iou': unet_result['std_iou']
            },
            'simple_rl': {
                'mean_dice': simple_rl_result['mean_dice'],
                'mean_iou': simple_rl_result['mean_iou'],
                'std_dice': simple_rl_result['std_dice'],
                'std_iou': simple_rl_result['std_iou'],
                'mean_steps': simple_rl_result['mean_steps']
            },
            'dice_difference': float(simple_rl_result['mean_dice'] - unet_result['mean_dice']),
            'iou_difference': float(simple_rl_result['mean_iou'] - unet_result['mean_iou']),
            'winner': 'simple_rl' if simple_rl_result['mean_dice'] > unet_result['mean_dice'] else 'unet'
        }
        
        # Save comparison to JSON file
        with open(os.path.join(results_dir, 'model_comparison.json'), 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Also save a human-readable summary
        with open(os.path.join(results_dir, 'model_comparison.txt'), 'w') as f:
            f.write(f'Model Comparison\n')
            f.write(f'===============\n\n')
            f.write(f'                U-Net      SimpleRL\n')
            f.write(f'Dice:           {unet_result["mean_dice"]:.4f} ± {unet_result["std_dice"]:.4f}      {simple_rl_result["mean_dice"]:.4f} ± {simple_rl_result["std_dice"]:.4f}\n')
            f.write(f'IoU:            {unet_result["mean_iou"]:.4f} ± {unet_result["std_iou"]:.4f}      {simple_rl_result["mean_iou"]:.4f} ± {simple_rl_result["std_iou"]:.4f}\n')
            f.write(f'\nDice Difference: {simple_rl_result["mean_dice"] - unet_result["mean_dice"]:.4f}')
            f.write(f'\nIoU Difference: {simple_rl_result["mean_iou"] - unet_result["mean_iou"]:.4f}')
            f.write(f'\nWinner based on Dice: {comparison["winner"].upper()}\n')
        
        # Generate comparison plots
        plt.figure(figsize=(10, 6))
        
        # Dice and IoU bar chart
        x = np.arange(2)
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        models = ['U-Net', 'SimpleRL']
        
        # Dice scores with error bars
        dice_means = [unet_result['mean_dice'], simple_rl_result['mean_dice']]
        dice_stds = [unet_result['std_dice'], simple_rl_result['std_dice']]
        
        # IoU scores with error bars
        iou_means = [unet_result['mean_iou'], simple_rl_result['mean_iou']]
        iou_stds = [unet_result['std_iou'], simple_rl_result['std_iou']]
        
        # Plot bars
        rects1 = ax.bar(x - width/2, dice_means, width, yerr=dice_stds, label='Dice', capsize=10)
        rects2 = ax.bar(x + width/2, iou_means, width, yerr=iou_stds, label='IoU', capsize=10)
        
        # Add labels
        ax.set_ylabel('Score')
        ax.set_title('Segmentation Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        plt.savefig(os.path.join(results_dir, 'performance_comparison.png'))
        plt.close()
        
        print(f'\nComparison results:')
        print(f'  U-Net Dice: {unet_result["mean_dice"]:.4f} ± {unet_result["std_dice"]:.4f}, IoU: {unet_result["mean_iou"]:.4f} ± {unet_result["std_iou"]:.4f}')
        print(f'  SimpleRL Dice: {simple_rl_result["mean_dice"]:.4f} ± {simple_rl_result["std_dice"]:.4f}, IoU: {simple_rl_result["mean_iou"]:.4f} ± {simple_rl_result["std_iou"]:.4f}')
        print(f'  Dice Difference: {simple_rl_result["mean_dice"] - unet_result["mean_dice"]:.4f}')
        print(f'  Winner: {comparison["winner"].upper()}')
    
    print(f"\n===== All operations completed =====")
    print(f"Results saved to: {results_dir}")
    print(f"Models saved to: {models_dir}")
    
    # Return results directory for potential further analysis
    return results_dir

if __name__ == '__main__':
    main() 