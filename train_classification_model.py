from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch
import os
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np
import copy
from src.UIElementClassifier import UIElementClassifier
import warnings
warnings.filterwarnings("ignore")

class UIElementsDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # Create a mapping for the labels
        self.label_mapping = {'AXStaticText': 0, 'AXButton': 1, 'AXImage': 2}
        self.dataframe['encoded_labels'] = self.dataframe['label'].map(self.label_mapping)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[index, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.dataframe.loc[index, 'encoded_labels']
        if self.transform:
            image = self.transform(image)
        return image, label

transformations = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = UIElementsDataset(csv_file='data/MAC_Classification/train/filtered_labels.csv', 
                                  img_dir='data/MAC_Classification/train', transform=transformations)
valid_dataset = UIElementsDataset(csv_file='data/MAC_Classification/validation/filtered_labels.csv', 
                                  img_dir='data/MAC_Classification/validation', transform=transformations)

labels = train_dataset.dataframe['encoded_labels']
class_weights = labels.value_counts().sort_index().apply(lambda x: 1.0 / x)
weights = labels.map(class_weights).to_numpy()
sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

train_loader = DataLoader(dataset=train_dataset, batch_size=64, sampler=sampler, num_workers=23, pin_memory=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False, num_workers=23, pin_memory=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UIElementClassifier(num_classes=3).to(device)

# Handling class imbalance !!
class_counts = train_dataset.dataframe['label'].value_counts().sort_index()
class_weights = 1. / class_counts
weights = class_weights[train_dataset.dataframe['label'].values]
sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

# Loss function with class weights
class_weights_tensor = torch.tensor(class_weights[train_dataset.dataframe['label'].unique()].values, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

from tqdm import tqdm
best_val_f1 = -np.inf
patience, epochs_no_improve = 15, 0
early_stop = False


def train_and_validate(epochs=50):
    global epochs_no_improve, best_val_f1, early_stop

    epoch_losses = []
    val_f1_scores = []
    previous_lr = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(enumerate(train_loader), desc='Epoch {:1d}'.format(epoch), leave=False, total=len(train_loader))
        for i, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(train_loss/((i+1)*train_loader.batch_size))})
        epoch_losses.append(train_loss/len(train_loader))
        # Validation phase
        model.eval()
        val_targets = []
        val_outputs = []
        with torch.no_grad():
            for images, labels in tqdm(valid_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_targets.extend(labels.tolist())
                val_outputs.extend(outputs.argmax(1).tolist())

        val_acc = accuracy_score(val_targets, val_outputs)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_targets, val_outputs, average='macro')
        val_f1_scores.append(val_f1)
        # Learning rate scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != previous_lr:
            print(f'\nLEARNING RATE changed to: {current_lr}')
            previous_lr = current_lr
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping triggered")
                early_stop = True
                break

        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
    return val_f1_scores, epoch_losses

val_f1_scores, epoch_losses = train_and_validate(epochs=100)

# plot

import matplotlib.pyplot as plt
plt.plot(epoch_losses)
plt.plot(val_f1_scores)
plt.title('Training Loss and Validation F1 Score')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation F1 Score'])
plt.savefig('training_plot_classification.png')
