import os
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, txt_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        with open(txt_file, 'r') as f:
            self.data_list = f.readlines()
            
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx].strip()
        img_name, label = line.split()
        
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, int(label)
