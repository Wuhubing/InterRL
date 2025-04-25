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
                        help='Mode of operation: train_unet (只训练U-Net), train_rl (训练PPO代理), train_simple_rl (训练InteractiveRL代理并保存训练历史), evaluate (评估), all (全部), compare_unet_simple_rl (比较UNet和SimpleRL), train_and_evaluate (训练并评估)')
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

def train_unet_model(args, device, data_loaders):
    """Train and evaluate the U-Net model with standardized outputs"""
    print('Training U-Net model...')
    
    # Create timestamp for this training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model output directories
    unet_model_dir = os.path.join('models', f'unet_{timestamp}')
    os.makedirs(unet_model_dir, exist_ok=True)
    
    # Create model
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    model = model.to(device)
    
    # Create optimizer and loss functions
    from src.unet_model import DiceLoss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce_criterion = nn.BCELoss()
    dice_criterion = DiceLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6, verbose=True
    )
    
    # Early stopping parameters with increased patience
    best_val_dice = 0.0
    best_epoch = 0
    patience = 50  # 增加早停的耐心值
    patience_counter = 0
    
    # Training history - 只存储在一个JSON文件中的训练历史
    history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'lr': [],
        'best_epoch': 0,
        'best_val_dice': 0.0,
        'best_val_iou': 0.0,
        'total_training_time': 0.0,
        'early_stopped': False
    }
    
    # Track overall training time
    training_start_time = time.time()
    
    for epoch in range(args.unet_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(data_loaders['train']):
            # Get images and masks
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss - combined BCE and Dice Loss
            bce_loss = bce_criterion(outputs, masks)
            dice_loss = dice_criterion(outputs, masks)
            loss = bce_loss + dice_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # 只在每10个批次打印一次进度
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.unet_epochs} | Batch {batch_idx}/{len(data_loaders['train'])} | Loss: {loss.item():.4f}")
        
        train_loss /= len(data_loaders['train'])
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loaders['val']):
                # Get images and masks
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                bce_loss = bce_criterion(outputs, masks)
                dice_loss = dice_criterion(outputs, masks)
                loss = bce_loss + dice_loss
                val_loss += loss.item()
                
                # Generate binary mask predictions with 0.5 threshold
                pred_masks = (outputs > 0.5).float()
                
                # Calculate batch metrics
                from src.unet_model import dice_coefficient, iou_score
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
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Record validation metrics
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_dice'].append(float(val_dice))
        history['val_iou'].append(float(val_iou))
        history['lr'].append(float(current_lr))
        
        # 保存所有训练历史到单个JSON文件
        with open(os.path.join(unet_model_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Epoch {epoch+1}/{args.unet_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Time: {epoch_time:.2f}s")
        
        # 只保存最好的模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_iou = val_iou
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Update history with best values
            history['best_epoch'] = best_epoch
            history['best_val_dice'] = float(best_val_dice)
            history['best_val_iou'] = float(best_val_iou)
            
            try:
                # 只保存一个最佳模型
                best_model_path = os.path.join(unet_model_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': val_loss,
                    'dice': val_dice,
                    'iou': val_iou
                }, best_model_path)
                
                # Also save a copy to the standard location
    torch.save(model.state_dict(), 'models/unet_model.pth')
    
                print(f"Saved new best U-Net model with validation Dice: {best_val_dice:.4f}")
            except Exception as e:
                print(f"Error saving best model: {str(e)}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
            history['early_stopped'] = True
            break
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    history['total_training_time'] = float(total_training_time)
    
    # If early stopped, load the best model for return
    if history['early_stopped'] or best_epoch < epoch + 1:
        print(f"Loading best model from epoch {best_epoch} for final evaluation")
        checkpoint = torch.load(os.path.join(unet_model_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save final training summary
    training_summary = {
        'model_type': 'unet',
        'num_epochs': epoch + 1,
        'total_epochs_possible': args.unet_epochs,
        'early_stopped': history['early_stopped'],
        'best_epoch': best_epoch,
        'best_val_dice': float(best_val_dice),
        'best_val_iou': float(best_val_iou),
        'final_train_loss': float(train_loss),
        'final_val_loss': float(val_loss),
        'final_val_dice': float(val_dice),
        'final_val_iou': float(val_iou),
        'total_training_time': float(total_training_time),
        'learning_rate': {
            'initial': args.lr,
            'final': float(current_lr)
        },
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'timestamp': timestamp
    }
    
    # 更新历史记录并保存
    history.update(training_summary)
    with open(os.path.join(unet_model_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # 生成训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['epochs'], history['train_loss'], label='Train Loss')
    plt.plot(history['epochs'], history['val_loss'], label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['epochs'], history['val_dice'], label='Dice')
    plt.plot(history['epochs'], history['val_iou'], label='IoU')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['epochs'], history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(unet_model_dir, 'training_curves.png'))
    plt.close()
    
    print(f"U-Net training completed in {total_training_time:.2f} seconds")
    print(f"Best validation Dice: {best_val_dice:.4f} at epoch {best_epoch}")
    print(f"Training artifacts saved to {unet_model_dir}")
    
    # Return the trained model and history
    return model, history

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
    from src.simple_rl import SimpleRLAgent, train_agent
    
    print('Training Simple RL agent...')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建代理
    agent = SimpleRLAgent(
        n_actions=7,  # 0=Up, 1=Down, 2=Left, 3=Right, 4=Expand, 5=Shrink, 6=Done
        lr=args.lr,
        gamma=0.99,
        device=device
    )
    
    # 直接使用传入的args参数
    train_agent(args)
    
    # 返回代理
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
    from src.simple_rl import predict_mask, visualize_results
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
            pred_mask, num_steps = predict_mask(agent, image, gt_mask, max_steps=max_steps, return_steps=True)
            inference_time = time.time() - start_time
            
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
    from src.simple_rl import SimpleRLAgent
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
            num_episodes=50
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
    # 导入必要的函数
    from src.unet_model import dice_coefficient, iou_score, DiceLoss
    from src.simple_rl import train_agent
    import numpy as np
    import pickle
    
    print('Training and evaluating models...')
    
    # 创建结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results/comparison_{timestamp}'
    models_dir = f'models/checkpoints_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, 'unet'), exist_ok=True)
    os.makedirs(os.path.join(models_dir, 'simple_rl'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'unet_data'), exist_ok=True)
    
    # 保存训练配置
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
    
    # 1. 训练U-Net模型并保存检查点
    print('\n1. Training U-Net model...')
    
    # 创建U-Net模型
    unet_model = UNet(n_channels=3, n_classes=1, bilinear=True)
    unet_model = unet_model.to(device)
    
    # 创建U-Net优化器
    optimizer = torch.optim.AdamW(unet_model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 使用组合损失函数：BCE + Dice Loss
    bce_criterion = nn.BCELoss()
    dice_criterion = DiceLoss()
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=True
    )
    
    # 用于早停和保存最佳模型
    best_val_dice = 0.0
    best_epoch = 0
    patience = 15  # 早停耐心值
    patience_counter = 0
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'lr': [],
        'epochs': [],
        'time_per_epoch': [],
        'batch_losses': [],  # 记录每个批次的损失
        'per_sample_dice': [],  # 每个样本的Dice分数
        'per_sample_iou': []    # 每个样本的IoU分数
    }
    
    for epoch in range(args.unet_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        unet_model.train()
        train_loss = 0
        batch_losses = []
        
        for batch_idx, batch in enumerate(data_loaders['train']):
            # 获取图像和掩码
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 正向传播
            outputs = unet_model(images)
            
            # 计算损失 - 组合BCE和Dice Loss
            bce_loss = bce_criterion(outputs, masks)
            dice_loss = dice_criterion(outputs, masks)
            loss = bce_loss + dice_loss
            
            # 记录每个批次的损失
            batch_losses.append(loss.item())
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet_model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            train_loss += loss.item()
            
            # 每10个批次打印一次进度
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.unet_epochs} | Batch {batch_idx}/{len(data_loaders['train'])} | Loss: {loss.item():.4f}")
        
        train_loss /= len(data_loaders['train'])
        history['train_loss'].append(train_loss)
        history['batch_losses'].append(batch_losses)
        
        # 验证阶段
        unet_model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        epoch_per_sample_dice = []
        epoch_per_sample_iou = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loaders['val']):
                # 获取图像和掩码
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # 正向传播
                outputs = unet_model(images)
                
                # 计算损失
                bce_loss = bce_criterion(outputs, masks)
                dice_loss = dice_criterion(outputs, masks)
                loss = bce_loss + dice_loss
                val_loss += loss.item()
                
                # 生成二值掩码预测，注意阈值是0.5
                pred_masks = (outputs > 0.5).float()
                
                # 逐样本计算指标
                for i in range(pred_masks.size(0)):
                    sample_dice = dice_coefficient(pred_masks[i:i+1], masks[i:i+1]).item()
                    sample_iou = iou_score(pred_masks[i:i+1], masks[i:i+1]).item()
                    
                    epoch_per_sample_dice.append(sample_dice)
                    epoch_per_sample_iou.append(sample_iou)
                
                # 计算批次级指标
                batch_dice = dice_coefficient(pred_masks, masks).item()
                batch_iou = iou_score(pred_masks, masks).item()
                
                val_dice += batch_dice
                val_iou += batch_iou
        
        val_loss /= len(data_loaders['val'])
        val_dice /= len(data_loaders['val'])
        val_iou /= len(data_loaders['val'])
        
        # 更新学习率调度器
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # 记录每个epoch的训练时间
        epoch_time = time.time() - epoch_start_time
        
        # 记录验证指标
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['lr'].append(current_lr)
        history['epochs'].append(epoch + 1)
        history['time_per_epoch'].append(epoch_time)
        history['per_sample_dice'].append(epoch_per_sample_dice)
        history['per_sample_iou'].append(epoch_per_sample_iou)
        
        print(f"Epoch {epoch+1}/{args.unet_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Time: {epoch_time:.2f}s")
        
        # 保存每个epoch的训练数据
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
        
        # 保存每个epoch的详细数据
        with open(os.path.join(results_dir, 'unet_data', f'epoch_{epoch+1}_data.pkl'), 'wb') as f:
            pickle.dump(epoch_data, f)
        
        # 可视化当前epoch的验证结果
        if args.save_visualizations:
            # 随机选择几个验证样本进行可视化
            with torch.no_grad():
                sample_indices = np.random.randint(0, len(data_loaders['val'].dataset), 3)
                
                for i, idx in enumerate(sample_indices):
                    sample = data_loaders['val'].dataset[idx]
                    image = sample['image'].unsqueeze(0).to(device)
                    gt_mask = sample['mask'].unsqueeze(0).to(device)
                    
                    output = unet_model(image)
                    pred_mask = (output > 0.5).float()
                    
                    # 计算指标
                    sample_dice = dice_coefficient(pred_mask, gt_mask).item()
                    sample_iou = iou_score(pred_mask, gt_mask).item()
                    
                    # 转换为numpy
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
        
        # 保存最佳模型，使用try-except块防止磁盘空间不足
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_path = os.path.join(models_dir, 'unet', 'unet_best_model.pth')
            
            # 使用try-except块保存最佳模型
            try:
                # 保存最佳模型
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': unet_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': val_loss,
                    'dice': val_dice,
                    'iou': val_iou
                }, best_model_path)
                
                print(f"保存新的最佳U-Net模型，验证Dice分数: {best_val_dice:.4f}")
            except Exception as e:
                print(f"保存最佳模型时出错: {str(e)}，跳过保存继续训练")
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= patience:
            print(f"早停在第{epoch+1}轮。{patience}轮内没有改进。")
            break
    
    # 保存最终模型，使用try-except块
    try:
        final_model_path = os.path.join(models_dir, 'unet', 'unet_final_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': unet_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_loss,
            'dice': val_dice,
            'iou': val_iou
        }, final_model_path)
    except Exception as e:
        print(f"保存最终模型时出错: {str(e)}，使用最佳模型继续")
    
    # 保存完整的训练历史记录
    with open(os.path.join(results_dir, 'unet_training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # 生成更多的训练可视化图表
    
    # 1. 损失与准确率曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['epochs'], history['train_loss'], label='Train Loss')
    plt.plot(history['epochs'], history['val_loss'], label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['epochs'], history['val_dice'], label='Dice')
    plt.plot(history['epochs'], history['val_iou'], label='IoU')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['epochs'], history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'unet_training_curves.png'))
    plt.close()
    
    # 2. 每个epoch的训练时间条形图
    plt.figure(figsize=(10, 5))
    plt.bar(history['epochs'], history['time_per_epoch'])
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'unet_training_time.png'))
    plt.close()
    
    # 3. 验证集上的分布图
    if len(history['per_sample_dice']) > 0:
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
        plt.savefig(os.path.join(results_dir, 'unet_score_distributions.png'))
        plt.close()
    
    # 加载最佳模型用于评估
    print(f"从第{best_epoch}轮加载最佳U-Net模型进行评估...")
    checkpoint = torch.load(best_model_path)
    unet_model.load_state_dict(checkpoint['model_state_dict'])
    unet_model.eval()
    print(f"加载了最佳U-Net模型，验证Dice: {checkpoint['dice']:.4f}, IoU: {checkpoint['iou']:.4f}")
    
    # 2. 训练SimpleRL模型
    print('\n2. Training SimpleRL model...')
    # 设置SimpleRL训练参数
    simple_rl_args = argparse.Namespace(
        data_dir=args.data_dir,
        output_dir=os.path.join(models_dir, 'simple_rl'),
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
        seed=42
    )
    
    # 直接调用simple_rl.py中的train_agent函数
    # 记录训练过程中的最佳性能
    train_start_time = time.time()
    simple_rl_training_results = train_agent(simple_rl_args)
    train_time = time.time() - train_start_time
    
    # 提取最佳验证性能（如果train_agent返回此信息）
    if isinstance(simple_rl_training_results, dict) and 'best_val_dice' in simple_rl_training_results:
        simple_rl_best_val_dice = simple_rl_training_results['best_val_dice']
        simple_rl_best_val_iou = simple_rl_training_results.get('best_val_iou', 0)
        simple_rl_best_episode = simple_rl_training_results.get('best_episode', 0)
        print(f'SimpleRL训练完成：最佳验证Dice={simple_rl_best_val_dice:.4f}, IoU={simple_rl_best_val_iou:.4f} at Episode {simple_rl_best_episode}')
    else:
        print('SimpleRL模型训练完成，但未返回最佳验证性能信息')
    
    print(f'SimpleRL模型训练用时：{train_time:.2f}秒')
    
    # 3. 加载模型进行评估
    print('\n3. Loading models for evaluation...')
    
    # U-Net模型已加载为最佳模型
    
    # 加载SimpleRL最佳模型
    simple_rl_agent = None
    from src.simple_rl import SimpleRLAgent
    try:
        print(f"正在加载SimpleRL最佳模型...")
        best_rl_path = os.path.join(models_dir, 'simple_rl', 'best_model.pth')
        
        # 检查最佳模型文件是否存在
        if os.path.isfile(best_rl_path):
            simple_rl_agent = SimpleRLAgent(
                n_actions=7,
                lr=args.lr,
                gamma=0.99,
                device=device
            )
            simple_rl_agent.load(best_rl_path)
            print('成功加载SimpleRL最佳模型')
            
            # 保存模型信息
            if isinstance(simple_rl_training_results, dict) and 'best_val_dice' in simple_rl_training_results:
                with open(os.path.join(results_dir, 'simple_rl_best_model_info.txt'), 'w') as f:
                    f.write(f'SimpleRL最佳模型信息\n')
                    f.write(f'===================\n\n')
                    f.write(f'Episode: {simple_rl_best_episode}\n')
                    f.write(f'验证Dice: {simple_rl_best_val_dice:.4f}\n')
                    f.write(f'验证IoU: {simple_rl_best_val_iou:.4f}\n')
                    f.write(f'模型路径: {best_rl_path}\n')
        else:
            print(f"警告：最佳模型文件不存在 ({best_rl_path})，尝试加载最终模型...")
            final_rl_path = os.path.join(models_dir, 'simple_rl', 'final_model.pth')
            if os.path.isfile(final_rl_path):
                simple_rl_agent = SimpleRLAgent(
                    n_actions=7,
                    lr=args.lr,
                    gamma=0.99,
                    device=device
                )
                simple_rl_agent.load(final_rl_path)
                print('成功加载SimpleRL最终模型')
            else:
                print(f"错误：最终模型文件也不存在 ({final_rl_path})")
                raise FileNotFoundError("找不到SimpleRL模型文件")
    except Exception as e:
        print(f'加载SimpleRL模型时出错: {str(e)}')
        print('SimpleRL模型无法加载，将跳过评估')
    
    # 4. 评估模型
    print('\n4. Evaluating models...')
    evaluation_results = {}
    
    # 评估U-Net
    print('Evaluating U-Net...')
    unet_results = evaluate_unet(
        model=unet_model,
        test_dataset=data_loaders['test'].dataset,
        device=device,
        save_dir=results_dir,
        num_samples=args.eval_samples,
        save_visualizations=args.save_visualizations
    )
    evaluation_results['unet'] = unet_results
    
    # 评估SimpleRL（如果可用）
    if simple_rl_agent:
        print('Evaluating SimpleRL...')
        # 在评估时使用与训练相同的max_steps参数
        simple_rl_results = evaluate_simple_rl(
            agent=simple_rl_agent,
            test_dataset=data_loaders['test'].dataset,
            device=device,
            max_steps=args.max_steps,
            save_dir=results_dir,
            num_samples=args.eval_samples,
            save_visualizations=args.save_visualizations
        )
        evaluation_results['simple_rl'] = simple_rl_results
        
        # 将训练验证集上的最佳性能也添加到评估结果中
        if isinstance(simple_rl_training_results, dict) and 'best_val_dice' in simple_rl_training_results:
            simple_rl_results['best_val_dice'] = simple_rl_training_results['best_val_dice']
            simple_rl_results['best_val_iou'] = simple_rl_training_results.get('best_val_iou', 0)
            simple_rl_results['best_episode'] = simple_rl_training_results.get('best_episode', 0)
    
    # 5. 对比结果
    if 'unet' in evaluation_results and 'simple_rl' in evaluation_results:
        print('\n5. Comparison of results:')
        print(f'                U-Net      SimpleRL')
        print(f'Dice:           {evaluation_results["unet"]["mean_dice"]:.4f}      {evaluation_results["simple_rl"]["mean_dice"]:.4f}')
        print(f'IoU:            {evaluation_results["unet"]["mean_iou"]:.4f}      {evaluation_results["simple_rl"]["mean_iou"]:.4f}')
        
        # 如果有训练验证集上的最佳性能，也打印出来
        if 'best_val_dice' in evaluation_results['simple_rl']:
            print(f'\nSimpleRL在验证集上的最佳性能（第{evaluation_results["simple_rl"]["best_episode"]}轮）:')
            print(f'验证Dice: {evaluation_results["simple_rl"]["best_val_dice"]:.4f}')
            print(f'验证IoU: {evaluation_results["simple_rl"]["best_val_iou"]:.4f}')
        
        # 保存对比结果到文件
        with open(os.path.join(results_dir, 'model_comparison.txt'), 'w') as f:
            f.write(f'Model Comparison\n')
            f.write(f'===============\n\n')
            f.write(f'                U-Net      SimpleRL\n')
            f.write(f'Dice:           {evaluation_results["unet"]["mean_dice"]:.4f}      {evaluation_results["simple_rl"]["mean_dice"]:.4f}\n')
            f.write(f'IoU:            {evaluation_results["unet"]["mean_iou"]:.4f}      {evaluation_results["simple_rl"]["mean_iou"]:.4f}\n')
            
            # 添加训练验证集上的最佳性能
            if 'best_val_dice' in evaluation_results['simple_rl']:
                f.write(f'\nSimpleRL在验证集上的最佳性能（第{evaluation_results["simple_rl"]["best_episode"]}轮）:\n')
                f.write(f'验证Dice: {evaluation_results["simple_rl"]["best_val_dice"]:.4f}\n')
                f.write(f'验证IoU: {evaluation_results["simple_rl"]["best_val_iou"]:.4f}\n')
            
        # 生成对比图
        plt.figure(figsize=(8, 6))
        models = ['U-Net', 'SimpleRL']
        dice_scores = [evaluation_results['unet']['mean_dice'], evaluation_results['simple_rl']['mean_dice']]
        iou_scores = [evaluation_results['unet']['mean_iou'], evaluation_results['simple_rl']['mean_iou']]
        
        # 添加SimpleRL验证集上的最佳性能
        if 'best_val_dice' in evaluation_results['simple_rl']:
            models.append('SimpleRL\n(验证集最佳)')
            dice_scores.append(evaluation_results['simple_rl']['best_val_dice'])
            iou_scores.append(evaluation_results['simple_rl']['best_val_iou'])
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, dice_scores, width, yerr=dice_stds, label='Dice', capsize=10)
        rects2 = ax.bar(x + width/2, iou_scores, width, yerr=iou_stds, label='IoU', capsize=10)
        
        ax.set_ylabel('Score')
        ax.set_title('Segmentation Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        # 添加数值标签
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
    
    return evaluation_results, models_dir, results_dir

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
        from src.simple_rl import train_agent, SimpleRLAgent
        
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
            from src.simple_rl import SimpleRLAgent
            
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