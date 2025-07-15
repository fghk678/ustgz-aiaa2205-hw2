#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the prediction for submission
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.vit import ViT_DANN, ViT
from dataset import CustomDataset
from tqdm import tqdm
import os

def predict_no_label_data(
    model_path='best_model_baseline.pth',
    predict_txt='data/image_list/target_to_predict.txt',
    data_dir='data',
    output_file='prediction.csv',
    batch_size=64,
    num_classes=65
):
    """
    Make predictions for data in target_to_predict.txt
    
    Args:
        model_path: Model checkpoint path
        predict_txt: Unlabeled data list file
        data_dir: Data directory
        output_file: Output prediction result file
        batch_size: Batch size
        num_classes: Number of classes
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data preprocessing
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and loader
    print(f"Loading data: {predict_txt}")
    dataset = CustomDataset(predict_txt, data_dir, transform=test_transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    print("Creating model...")
    # model = ViT_DANN(num_classes=num_classes, pretrained=False).to(device)
    model = ViT(num_classes=num_classes, pretrained=False).to(device)
    
    # Load model weights
    print(f"Loading model weights: {model_path}")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        # model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint)
        print(f"Model loaded successfully, best accuracy: {checkpoint.get('best_acc', 'N/A')}")
    else:
        print(f"Error: Model file {model_path} not found")
        return
    
    # Start prediction
    print("Starting prediction...")
    model.eval()
    predictions = []
    filenames = []
    
    # Read filename list
    with open(predict_txt, 'r') as f:
        file_lines = f.readlines()
    filenames = [line.strip().split()[0] for line in file_lines]
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(data_loader, desc="Prediction progress")):
            x = x.to(device)
            # logits, _ = model(x, alpha=0.0)  
            logits = model(x)
            preds = logits.argmax(1)
            predictions.extend(preds.cpu().numpy())
    
    # Save prediction results to CSV file
    print(f"Saving prediction results to: {output_file}")
    with open(output_file, 'w') as f:
        # Write CSV header
        f.write("Id,Category\n")
        
        # Write prediction results
        for filename, pred in zip(filenames, predictions):
            # Extract filename (remove path prefix)
            base_name = filename.split('/')[-1]
            f.write(f"{base_name},{pred}\n")
    
    print(f"Prediction completed! Processed {len(predictions)} samples")
    print(f"Prediction results saved to: {output_file}")
    

if __name__ == '__main__':
    # Prediction parameters
    MODEL_PATH = 'best_model_baseline.pth'
    PREDICT_TXT = 'data/image_list/target_to_predict.txt'
    DATA_DIR = 'data'
    OUTPUT_FILE = 'prediction.csv'
    
    # Execute prediction
    predict_no_label_data(
        model_path=MODEL_PATH,
        predict_txt=PREDICT_TXT,
        data_dir=DATA_DIR,
        output_file=OUTPUT_FILE,
        batch_size=64,
        num_classes=65
    ) 