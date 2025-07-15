import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.vit import ViT_DANN
from dataset import CustomDataset
from utils import set_seed, accuracy
from tqdm import tqdm
import os

def train_domain_adaptation(
    src_txt, src_dir, tgt_txt, tgt_eval_txt, tgt_dir, num_classes=65, batch_size=32, epochs=20, alpha=0.3
):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create checkpoint directory
    os.makedirs('ckpt', exist_ok=True)

    # Data transforms
    # ==== TODO: Try advanced data augmentation techniques here! ====
    # Example: transforms.ColorJitter, transforms.RandomGrayscale, RandAugment, Mixup, CutMix, etc.
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and Loaders
    src_dataset = CustomDataset(src_txt, src_dir, transform=train_transform)
    tgt_dataset = CustomDataset(tgt_txt, tgt_dir, transform=test_transform)
    tgt_eval_dataset = CustomDataset(tgt_eval_txt, tgt_dir, transform=test_transform)
    src_loader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    tgt_loader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    # Create validation data loader
    tgt_val_loader = DataLoader(tgt_eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ViT_DANN(num_classes=num_classes, pretrained=True).to(device)
    
    # ==== TODO: Try differential learning rates for different parts of the model ====
    # e.g. backbone(vit): 1e-5, head: 5e-5
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    
    
    for epoch in range(epochs):
        model.train()
        total_cls_loss = 0
        total_dom_loss = 0
        correct, total = 0, 0

        # Make src_loader and tgt_loader the same length
        tgt_iter = iter(tgt_loader)
        for batch_idx, (src_x, src_y) in enumerate(tqdm(src_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            try:
                tgt_x, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_x, _ = next(tgt_iter)

            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)
            bs = src_x.size(0)

            # 1. classification output and domain output
            src_cls_logits, src_dom_logits = model(src_x, alpha=alpha)
            _, tgt_dom_logits = model(tgt_x, alpha=alpha)

            # 2. source domain classification loss
            cls_loss = cls_criterion(src_cls_logits, src_y)

            # 3. domain classification loss
            dom_labels = torch.cat([
                torch.zeros(bs, dtype=torch.long),
                torch.ones(bs, dtype=torch.long)
            ]).to(device)
            dom_logits = torch.cat([src_dom_logits, tgt_dom_logits], dim=0)
            dom_loss = domain_criterion(dom_logits, dom_labels)

            # 4. total loss
            # ==== TODO: Tune the weighting factor for domain loss (gamma) ====
            # Try different values (e.g., 0.1~1.0) to balance classification and domain adaptation.
            loss = cls_loss + 0.2*dom_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_cls_loss += cls_loss.item()
            total_dom_loss += dom_loss.item()

            pred = src_cls_logits.argmax(1)
            correct += (pred == src_y).sum().item()
            total += bs

        scheduler.step()
        
        # Calculate training metrics
        train_acc = 100 * correct / total
        avg_cls_loss = total_cls_loss / len(src_loader)
        avg_dom_loss = total_dom_loss / len(src_loader)
        
        print(f"Epoch {epoch+1}: ClsLoss={avg_cls_loss:.4f}, "
              f"DomLoss={avg_dom_loss:.4f}, "
              f"Train Acc={train_acc:.2f}%")
        
        # Validation phase
        if epoch % 5 == 0 or epoch == epochs - 1:  # Validate every 5 epochs, also validate the last epoch
            val_acc = eval_model(model, tgt_val_loader, device)

            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                    'cls_loss': avg_cls_loss,
                    'dom_loss': avg_dom_loss,
                }, 'ckpt/best_model.pth')
                print(f"Saved best model with validation accuracy: {best_acc:.2f}%")
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'cls_loss': avg_cls_loss,
            'dom_loss': avg_dom_loss,
        }, 'ckpt/latest_model.pth')
    
    print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
    return model, best_acc

def eval_model(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in tqdm(data_loader, desc='Evaluating'):
            x, y = x.to(device), y.to(device)
            logits, _ = model(x, alpha=0.0)  # 验证时不使用梯度反转
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            preds = logits.argmax(1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = accuracy(all_preds, all_labels)
    avg_loss = total_loss / len(data_loader)
    
    print(f"Eval Loss: {avg_loss:.4f}, Eval Accuracy: {100*acc:.2f}%")
    return acc

def load_model(checkpoint_path, model, optimizer=None, scheduler=None):
    """load checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best accuracy: {100*checkpoint['best_acc']:.2f}%")
    
    return checkpoint['epoch'], checkpoint['best_acc']

if __name__ == '__main__':
    # Modify to your data path
    SRC_TXT = 'data/image_list/source.txt'
    SRC_DIR = 'data'
    TGT_TXT = 'data/image_list/target.txt'
    TGT_DIR = 'data'  
    TGT_EVAL_TXT = 'data/image_list/target_eval.txt'

    # Train model
    model, best_acc = train_domain_adaptation(
        SRC_TXT, SRC_DIR, TGT_TXT, TGT_EVAL_TXT, TGT_DIR,
        num_classes=65, batch_size=32, epochs=30, alpha=0.3
    )
    
    print(f"\n=== Training Summary ===")
    print(f"Best validation accuracy: {100*best_acc:.2f}%")
