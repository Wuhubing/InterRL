import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Standard U-Net architecture - don't easily change the channel numbers to avoid dimension mismatch
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        self.outc = OutConv(64, n_classes)
        
        # Add Dropout to reduce overfitting
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply dropout at the deepest layer
        x5 = self.dropout(x5)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    """Combined Binary Cross Entropy and Dice Loss"""
    def __init__(self, smooth=1.0, weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.weight = weight  # Weight between Dice and BCE
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth)
        
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        # Weighted sum
        return self.weight * bce_loss + (1 - self.weight) * dice_loss

def train_unet(model, train_loader, val_loader, device, epochs=50, lr=1e-4, patience=10):
    """
    Train U-Net model with early stopping and learning rate scheduling
    
    Args:
        model (nn.Module): U-Net model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to train on
        epochs (int): Number of epochs
        lr (float): Learning rate
        patience (int): Early stopping patience
        
    Returns:
        dict: Training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Use learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=patience//2, verbose=True
    )
    
    # Use combined loss function
    criterion = DiceBCELoss(weight=0.5)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'early_stopped': False,
        'best_epoch': 0,
        'time_per_epoch': [],
        'per_sample_dice': [],
        'per_sample_iou': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        import time
        epoch_start = time.time()
        
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            # Get inputs and targets
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        
        batch_dices = []
        batch_ious = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Get inputs and targets
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Forward pass
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                
                # Calculate metrics
                pred_masks = (outputs > 0.5).float()
                dice = dice_coefficient(pred_masks, masks)
                iou = iou_score(pred_masks, masks)
                
                # Record performance of each sample
                for i in range(masks.size(0)):
                    d = dice_coefficient(pred_masks[i:i+1], masks[i:i+1]).item()
                    io = iou_score(pred_masks[i:i+1], masks[i:i+1]).item()
                    batch_dices.append(d)
                    batch_ious.append(io)
                
                val_dice += dice.item()
                val_iou += iou.item()
        
        # Calculate average losses and metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        
        # Record epoch time
        epoch_time = time.time() - epoch_start
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['time_per_epoch'].append(epoch_time)
        history['per_sample_dice'].append(batch_dices)
        history['per_sample_iou'].append(batch_ious)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Time: {epoch_time:.2f}s')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
            history['best_epoch'] = epoch
            print(f"Saving new best model, validation loss: {val_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"Validation loss not improved, counter: {early_stop_counter}/{patience}")
            
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                history['early_stopped'] = True
                break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model (Epoch {history['best_epoch']+1})")
    
    return history

def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """
    Calculate Dice Coefficient
    
    Args:
        y_pred (torch.Tensor): Predicted masks
        y_true (torch.Tensor): Ground truth masks
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        torch.Tensor: Dice coefficient
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    intersection = (y_pred * y_true).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    
    return dice

def iou_score(y_pred, y_true, smooth=1e-6):
    """
    Calculate IoU (Intersection over Union)
    
    Args:
        y_pred (torch.Tensor): Predicted masks
        y_true (torch.Tensor): Ground truth masks
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        torch.Tensor: IoU score
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou 