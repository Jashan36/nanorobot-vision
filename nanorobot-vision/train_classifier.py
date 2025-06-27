# train_instance_classifier.py
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from scipy.ndimage import label, find_objects
import argparse
from nucleus_classifier import NucleusClassifier
import kagglehub

# --- Step 1: Data Pre-processing ---
def create_cropped_dataset(raw_dataset_path: str, output_dir: str, class_map: Dict, crop_size: int = 48):
    """
    Processes a raw instance-level dataset into a new dataset of cropped nuclei,
    sorted into class folders for training.
    """
    raw_path, out_path = Path(raw_dataset_path), Path(output_dir)
    if out_path.exists():
        print(f"Cropped dataset directory '{output_dir}' already exists. Skipping pre-processing.")
        return
    out_path.mkdir(parents=True, exist_ok=True)
    
    csv_file = pd.read_csv(raw_path / 'Classification_labels.csv')
    print("Pre-processing raw data into cropped nuclei dataset...")

    for _, row in tqdm(csv_file.iterrows(), total=len(csv_file), desc="Processing Images"):
        img_name = row['Image_name']
        image = cv2.imread(str(raw_path / 'Images' / f"{img_name}.png"), cv2.IMREAD_COLOR)
        mask = np.array(Image.open(raw_path / 'Masks' / f"{img_name}_mask.png"))
        if image is None or mask is None: continue

        labeled_mask, _ = label(mask)
        objects = find_objects(labeled_mask)

        for i, obj_slice in enumerate(objects):
            (y_slice, x_slice) = obj_slice
            cy, cx = (y_slice.start + y_slice.stop) // 2, (x_slice.start + x_slice.stop) // 2
            
            class_id = mask[cy, cx]
            if class_id not in class_map: continue
            class_name = class_map[class_id]
            
            y1, y2 = max(0, cy - crop_size // 2), min(image.shape[0], cy + crop_size // 2)
            x1, x2 = max(0, cx - crop_size // 2), min(image.shape[1], cx + crop_size // 2)

            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                class_dir = out_path / class_name
                class_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(class_dir / f"{img_name}_{i}.png"), crop)

# --- Step 2: Training ---
def train_model(cropped_dataset_path: str, output_model_path: str, num_classes: int, epochs: int, batch_size: int, lr: float):
    """Initializes and trains the NucleusClassifier."""
    # Initialize a new, untrained classifier
    classifier = NucleusClassifier(model_path=None, num_classes=num_classes)
    
    # Start the training process
    classifier.train(
        dataset_path=cropped_dataset_path,
        output_model_path=output_model_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an Instance Classifier for Nuclei.")
    parser.add_argument('--dataset', required=True, help="Kaggle dataset handle (e.g., 'andrewmvd/cancer-instance-segmentation-and-classification')")
    parser.add_argument('--output-model', '-o', default='nucleus_classifier_model.pth', help='Path to save the best trained model.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    args = parser.parse_args()

    # --- Dataset-specific Configurations ---
    DATASET_CONFIGS = {
        'andrewmvd/cancer-instance-segmentation-and-classification': {
            "num_classes": 3,
            "class_map": {1: 'Neoplastic', 2: 'Inflammatory', 5: 'Epithelial'}, # Ignore Connective and Dead cells
            "class_names": ['Epithelial', 'Inflammatory', 'Neoplastic'] # Alphabetical order from folders
        },
        'andrewmvd/cancer-instance-segmentation-and-classification-2': {
            "num_classes": 5,
            "class_map": {1: 'Neoplastic', 2: 'Inflammatory', 3: 'Connective', 4: 'Dead', 5: 'Epithelial'},
            "class_names": ['Connective', 'Dead', 'Epithelial', 'Inflammatory', 'Neoplastic']
        },
        'andrewmvd/cancer-instance-segmentation-and-classification-3': {
            "num_classes": 2,
            "class_map": {1: 'Benign', 2: 'Malignant'},
            "class_names": ['Benign', 'Malignant']
        }
    }
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{args.dataset}' is not supported. Please add its config.")
        
    config = DATASET_CONFIGS[args.dataset]
    
    print(f"Downloading dataset '{args.dataset}'...")
    raw_data_path = kagglehub.dataset_download(args.dataset)
    
    cropped_data_dir = f"./cropped_{Path(raw_data_path).name}"

    print("\n--- STAGE 1: Pre-processing ---")
    create_cropped_dataset(raw_data_path, cropped_data_dir, config['class_map'])
    
    print("\n--- STAGE 2: Training Model ---")
    train_model(
        cropped_dataset_path=cropped_data_dir,
        output_model_path=args.output_model,
        num_classes=config['num_classes'],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    print(f"\nTraining complete. Model saved to '{args.output_model}'.")
