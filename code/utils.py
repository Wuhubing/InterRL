import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage.transform import resize

def dice_coefficient(pred, target, eps=1e-6):
    """
    Compute Dice coefficient between predicted mask and target mask
    
    Args:
        pred: Predicted mask
        target: Target mask
        eps: Small value to avoid division by zero
        
    Returns:
        Dice coefficient
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Flatten arrays
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Binarize if needed
    if pred_flat.dtype != bool:
        pred_flat = pred_flat > 0.5
    if target_flat.dtype != bool:
        target_flat = target_flat > 0.5
    
    # Compute intersection and sums
    intersection = np.sum(pred_flat & target_flat)
    pred_sum = np.sum(pred_flat)
    target_sum = np.sum(target_flat)
    
    # Compute Dice
    dice = (2. * intersection + eps) / (pred_sum + target_sum + eps)
    
    return dice

def iou_score(pred, target, eps=1e-6):
    """
    Compute IoU score between predicted mask and target mask
    
    Args:
        pred: Predicted mask
        target: Target mask
        eps: Small value to avoid division by zero
        
    Returns:
        IoU score
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Flatten arrays
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Binarize if needed
    if pred_flat.dtype != bool:
        pred_flat = pred_flat > 0.5
    if target_flat.dtype != bool:
        target_flat = target_flat > 0.5
    
    # Compute intersection and union
    intersection = np.sum(pred_flat & target_flat)
    union = np.sum(pred_flat | target_flat)
    
    # Compute IoU
    iou = (intersection + eps) / (union + eps)
    
    return iou

def plot_losses(train_losses, val_losses=None, path=None):
    """
    Plot training and validation losses
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def save_image(image, path, normalize=False):
    """
    Save image to disk
    
    Args:
        image: Image to save
        path: Path to save the image
        normalize: Whether to normalize the image
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # Convert from CHW to HWC format if needed
    if len(image.shape) == 3 and image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))
    
    # Squeeze single-channel images
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    
    # Normalize if requested
    if normalize:
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Ensure values are in [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Save image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(image).save(path)

def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 