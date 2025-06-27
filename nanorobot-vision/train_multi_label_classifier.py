# train_multi_label_classifier.py
import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import kagglehub

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from multi_label_classifier import MultiLabelClassifier

# --- KEY DATA LOADING CHANGE ---
# We need a custom Dataset class because we can't use ImageFolder.
class HPADataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform: transforms.Compose = None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = row['Id']
        # The tiles are split across multiple color channels/folders (Yellow, Red, Blue, Green)
        # We will use the yellow channel as the primary image for this example
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Handle cases where image might not exist, return a dummy
            print(f"Warning: Could not find image {img_path}. Returning dummy data.")
            image = Image.new("RGB", (256, 256))
            
        # The 'Target' column contains space-separated label indices
        label_indices = [int(i) for i in row['Target'].split()]
        
        # Create a one-hot encoded vector for the multi-label target
        target = torch.zeros(MultiLabelClassifier.NUM_CLASSES)
        for index in label_indices:
            # Check if index is valid for our trimmed class list
            if 0 <= index < MultiLabelClassifier.NUM_CLASSES:
                target[index] = 1.0

        if self.transform:
            image = self.transform(image)
        
        return image, target


def main():
    parser = argparse.ArgumentParser(description="Train a Multi-Label Classifier for Human Protein Atlas.")
    parser.add_argument('--output-model', '-o', default='hpa_model.pth', help='Path to save the trained model.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    args = parser.parse_args()

    # --- Step 1: Download and Prepare Data ---
    print("Downloading Human Protein Atlas tile datasets...")
    yellow_path = kagglehub.dataset_download("dschettler8845/human-protein-atlas-yellow-cell-tile-dataset")
    # Red, Blue, Green channels can also be downloaded and combined for a 4-channel input,
    # but we will stick to a simpler 3-channel (RGB) approach for now using one color.
    
    # The CSV with labels is typically in the base dataset, not the tile sets.
    # We will need the main HPA dataset for the train.csv file.
    main_hpa_path = kagglehub.dataset_download("pasanw/human-protein-atlas-2019-256x256")
    
    train_df = pd.read_csv(os.path.join(main_hpa_path, 'train.csv'))
    
    # We'll train on the 'yellow' tiles for this example.
    image_directory = os.path.join(yellow_path, 'train')
    
    # --- Step 2: Create Custom PyTorch Datasets ---
    print("Creating datasets and dataloaders...")
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = HPADataset(df=train_df, img_dir=image_directory)
    
    # Split the dataset
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- Step 3: Initialize and Train the Model ---
    classifier = MultiLabelClassifier(model_path=None)
    
    classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        output_model_path=args.output_model,
        epochs=args.epochs,
        lr=args.lr
    )
    
    print(f"\nTraining complete. Model saved to '{args.output_model}'")


if __name__ == "__main__":
    # Scikit-learn is a new dependency for this script for the F1 score.
    # Add 'scikit-learn' to your requirements.txt
    from sklearn.metrics import f1_score
    import numpy as np
    main()