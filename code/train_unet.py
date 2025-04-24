import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys

# Add code directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.data_utils import PolypDataset, get_data_loaders, visualize_sample
from code.unet_model import UNet, train_unet, dice_coefficient, iou_score

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set data directory
    data_dir = 'data/raw'
    
    # Get data loaders
    data_loaders = get_data_loaders(data_dir, batch_size=4, num_workers=2, seed=42)
    
    # Create model
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    
    # Train model
    history = train_unet(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        device=device,
        epochs=30,
        lr=1e-4
    )
    
    # Plot training history
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_dice'], label='Dice')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_iou'], label='IoU')
    plt.title('IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/unet_training_history.png')
    plt.close()
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/unet_model.pth')
    
    # Evaluate model on test set
    model.eval()
    test_dice = 0
    test_iou = 0
    
    with torch.no_grad():
        for batch in data_loaders['test']:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            pred_masks = (outputs > 0.5).float()
            
            test_dice += dice_coefficient(pred_masks, masks).item()
            test_iou += iou_score(pred_masks, masks).item()
    
    test_dice /= len(data_loaders['test'])
    test_iou /= len(data_loaders['test'])
    
    print(f'Test Dice: {test_dice:.4f}, Test IoU: {test_iou:.4f}')
    
    # Visualize some predictions
    os.makedirs('results', exist_ok=True)
    
    model.eval()
    test_dataset = data_loaders['test'].dataset
    
    fig, axes = plt.subplots(5, 3, figsize=(15, 25))
    
    for i in range(min(5, len(test_dataset))):
        # Get a sample from the test set
        sample = test_dataset[i]
        
        # Get image and mask
        image = sample['image'].unsqueeze(0).to(device)
        mask = sample['mask']
        
        # Get prediction
        with torch.no_grad():
            output = model(image)
            pred_mask = (output > 0.5).float().squeeze().cpu().numpy()
        
        # Convert tensors to numpy for visualization
        display_image = sample['image'].permute(1, 2, 0).cpu().numpy()
        if display_image.max() <= 1.0:
            display_image = (display_image * 255).astype(np.uint8)
            
        display_mask = sample['mask'].squeeze().cpu().numpy()
        
        # Plot original image
        axes[i, 0].imshow(display_image)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Plot ground truth mask
        axes[i, 1].imshow(display_mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Plot predicted mask
        axes[i, 2].imshow(pred_mask, cmap='gray')
        dice_val = dice_coefficient(torch.tensor(pred_mask), torch.tensor(display_mask))
        axes[i, 2].set_title(f'Prediction (Dice: {dice_val:.4f})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/unet_predictions.png')
    plt.close()
    
if __name__ == '__main__':
    main() 