# ==== Assignment Instructions (For students, do not modify) ====
"""
Assignment: Vision Transformer (ViT) Fine-Tuning & Domain Adaptation

1. Complete all # TODO sections in the code:
   - Implement the custom dataset
   - Replace the ViT classifier head
   - Complete the training and validation loops

2. Experiment with different data augmentation strategies and report your findings.

3. (Recommended) Try both single-domain and cross-domain (domain adaptation) training,
   and compare validation accuracies.

How to run:
    python vit_finetuning_starter.py

If you have questions, feel free to ask the instructor or TA.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import numpy as np

# ==== 1. Custom Dataset (TODO) ====
class CustomDataset(Dataset):
    def __init__(self, txt_file, data_dir, transform=None, max_samples=None):
        self.data_dir = data_dir
        self.transform = transform
        # TODO: Read all lines from txt_file and store in self.data_list
        # Each line: image_filename label
        # Example: self.data_list = ...
        
        # TODO: Only keep the first max_samples samples if specified
        pass

    def __len__(self):
        # TODO: Return the number of samples
        # Example: return len(self.data_list)
        pass

    def __getitem__(self, idx):
        # TODO:
        # 1. Get image filename and label from self.data_list
        # 2. Load the image (RGB)
        # 3. Apply self.transform if provided
        # 4. Return the image tensor and label as int
        pass

# ==== 2. Model Setup (TODO) ====
class ViT(nn.Module):
    def __init__(self, num_classes=65, pretrained=True):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        # TODO: Replace the ViT classifier head with nn.Linear for num_classes
        # Example: self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)
        pass

    def forward(self, x):
        return self.vit(x)

# ==== 3. Training Function (TODO: Complete training steps) ====
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        # TODO: Move images and labels to device
        # images, labels = ..., ...
        # TODO: Zero optimizer gradients
        # TODO: Forward pass, compute loss
        # outputs = model(images)
        # loss = criterion(outputs, labels)
        # TODO: Backward pass, optimizer step
        # loss.backward()
        # optimizer.step()
        # TODO: Accumulate loss and compute accuracy
        pass

    # TODO: Return average loss and accuracy
    # return total_loss / len(train_loader), 100. * correct / total
    pass

# ==== 4. Validation Function (TODO: Complete validation steps) ====
@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(val_loader, desc='Validating'):
        # TODO: Move images and labels to device
        # outputs = model(images)
        # loss = criterion(outputs, labels)
        # TODO: Accumulate loss and compute accuracy
        pass

    # TODO: Return average loss and accuracy
    # return total_loss / len(val_loader), 100. * correct / total
    pass

# ==== 5. Domain Adaptation Baseline ====
def domain_adaptation_baseline():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: Design/train data augmentation pipeline (try adding more methods)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # TODO: Add more augmentations here if you wish
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    source_dataset = CustomDataset(
        txt_file='data/image_list/source.txt',
        data_dir='data',
        transform=train_transform
    )
    target_dataset = CustomDataset(
        txt_file='data/image_list/target.txt',
        data_dir='data',
        transform=val_transform
    )

    source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True, num_workers=4)
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataset = CustomDataset(
        txt_file='data/image_list/target_eval.txt',
        data_dir='data',
        transform=val_transform
    )
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = ViT(num_classes=65, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_acc = 0
    for epoch in range(50):
        print(f"\nEpoch {epoch+1}/50")
        # TODO: Call train_epoch and validate functions, get training/validation results
        # train_loss, train_acc = ...
        # val_loss, val_acc = ...
        # TODO: Step the scheduler
        scheduler.step()

        # TODO: Save the best model based on validation accuracy
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     torch.save(model.state_dict(), 'best_model_baseline.pth')
        #     print(f"Saved best model with val_acc: {val_acc:.2f}%")

        # TODO: Print training and validation results
        # print(...)
        pass

def main():
    domain_adaptation_baseline()
    

if __name__ == '__main__':
    main()

    # Generate the prediction for submission
    MODEL_PATH = 'best_model_baseline.pth' # change to your trained model path
    PREDICT_TXT = 'data/image_list/target_to_predict.txt'
    DATA_DIR = 'data'
    OUTPUT_FILE = 'prediction.txt'
    
    from predict import predict_no_label_data
    
    predict_no_label_data(
        model_path=MODEL_PATH,
        predict_txt=PREDICT_TXT,
        data_dir=DATA_DIR,
        output_file=OUTPUT_FILE,
        batch_size=32,
        num_classes=65
    ) 
