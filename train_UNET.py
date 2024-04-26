# %%
from src.UNET import UNet
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# %%

class SegmentationDataset(Dataset):
    def __init__(self, directory, image_transform=None, mask_transform=None):
        self.directory = directory
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = [os.path.join(directory, x) for x in os.listdir(directory) if x.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = img_path.replace('.jpg', '.png')
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
    

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


mask_transform = transforms.Compose([
    transforms.ToTensor()
])


train_dataset = SegmentationDataset('data/roboflow_modified/train', image_transform=image_transform, mask_transform=mask_transform)
val_dataset = SegmentationDataset('data/roboflow_modified/valid', image_transform=image_transform, mask_transform=mask_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=23, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=23, pin_memory=True)




# %%
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_dice_max = -np.Inf
        self.delta = delta

    def __call__(self, val_dice, model):
        score = val_dice

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_dice, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_dice, model)
            self.counter = 0

    def save_checkpoint(self, val_dice, model):
        if self.verbose:
            print(f'Validation Dice Score increased ({self.val_dice_max:.6f} --> {val_dice:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_dice_max = val_dice

# %%
model = UNet(n_channels=3, n_classes=1, bilinear=True)
model = model.cuda()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.BCEWithLogitsLoss()
def dice_loss(pred, target, smooth = 1.):
    pred = torch.sigmoid(pred)
    numerator = 2 * torch.sum(pred * target) + smooth
    denominator = torch.sum(pred + target) + smooth
    return 1 - numerator / denominator
def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()  # Convert to binary
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)



early_stopping = EarlyStopping(patience=5, verbose=True)


# %%
from tqdm import tqdm
def plot_metrics(train_losses, val_dice_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_dice_scores, label='Validation Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.title('Training Loss and Validation Dice Score Over Time')
    plt.legend()
    plt.savefig('metrics.png')
    # plt.show()

    
def train_loop(num_epochs=50):
    train_losses = []
    val_dice_scores = []
    for epoch in range(num_epochs): 
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch+1}')

        for i, (images, masks) in progress_bar:
            images, masks = images.cuda(), masks.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({'Loss': (running_loss/(i+1))/8})

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1} Train loss: {epoch_loss:.4f}')

        # Validation using Dice Score
        model.eval()
        val_running_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.cuda(), masks.cuda()
                outputs = model(images)
                val_dice = dice_coefficient(outputs, masks)
                val_running_dice += val_dice.item() * images.size(0)

        val_epoch_dice = val_running_dice / len(val_loader.dataset)
        val_dice_scores.append(val_epoch_dice)
        print(f'Epoch {epoch+1} Validation Dice: {val_epoch_dice:.4f}')

        # LR scheduling
        scheduler.step()
        print('LR:', scheduler.get_last_lr())
        # Early Stopping
        early_stopping(val_epoch_dice, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        plot_metrics(train_losses, val_dice_scores)
    return train_losses, val_dice_scores

train_losses, val_dice_scores = train_loop()

# Plotting training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_dice_scores, label='Validation Dice')
plt.xlabel('Epochs')
plt.ylabel('Metric')
plt.title('Training Loss and Validation Dice Score Over Time')
plt.legend()
plt.show()


