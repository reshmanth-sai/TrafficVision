import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

CLASSES = ['bus', 'car', 'motorcycle', 'truck']

class VehicleMultiLabelDataset(Dataset):
    """
    Dataset loader for multi-label vehicle scene classification.
    Reads images and binary multi-label vectors [bus, car, motorcycle, truck]
    from a Roboflow classification export _classes.csv file.
    """
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        df = pd.read_csv(csv_file)
        # Clean column names (strip leading/trailing whitespace)
        df.columns = [c.strip() for c in df.columns]
        
        self.df = df
        self.filenames = df['filename'].tolist()
        
        # Verify required classes exist in CSV
        for col in CLASSES:
            if col not in df.columns:
                raise ValueError(f"Missing required class column '{col}' in {csv_file}")
        
        # Extract target labels as float32 array
        self.labels = df[CLASSES].astype('float32').values

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = str(self.filenames[idx]).strip()
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Error loading image at {img_path}: {e}")
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_name


def get_transforms(img_size=224):
    """
    Returns image transformation pipelines for training and evaluation.
    Standard ImageNet normalization mean and std.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform
