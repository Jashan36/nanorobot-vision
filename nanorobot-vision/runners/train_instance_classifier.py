# runners/train_instance_classifier.py
import os, cv2, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from scipy.ndimage import label, find_objects
import kagglehub
from core.nucleus_classifier import NucleusClassifier

def create_cropped_dataset(raw_dataset_path: str, output_dir: str, class_map: dict):
    raw_path, out_path = Path(raw_dataset_path), Path(output_dir)
    if out_path.exists():
        print(f"Cropped dataset at '{output_dir}' already exists. Skipping pre-processing.")
        return
    out_path.mkdir(parents=True, exist_ok=True)
    
    csv_file = pd.read_csv(raw_path / 'Classification_labels.csv')
    print("Pre-processing raw data into cropped nuclei dataset...")

    for _, row in tqdm(csv_file.iterrows(), total=len(csv_file), desc="Cropping Nuclei"):
        img_name, crop_size = row['Image_name'], 48
        image = cv2.imread(str(raw_path / 'Images' / f"{img_name}.png"))
        mask = np.array(Image.open(raw_path / 'Masks' / f"{img_name}_mask.png"))
        if image is None or mask is None: continue

        labeled_mask, _ = label(mask)
        objects = find_objects(labeled_mask)
        for i, obj_slice in enumerate(objects):
            (y_slice, x_slice) = obj_slice
            cy, cx = (y_slice.start + y_slice.stop) // 2, (x_slice.start + x_slice.stop) // 2
            class_id = mask[cy, cx]
            if class_id not in class_map: continue
            
            y1, y2 = max(0, cy - crop_size//2), min(image.shape[0], cy + crop_size//2)
            x1, x2 = max(0, cx - crop_size//2), min(image.shape[1], cx + crop_size//2)
            crop = image[y1:y2, x1:x2]

            if crop.size > 0:
                class_dir = out_path / class_map[class_id]
                class_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(class_dir / f"{img_name}_{i}.png"), crop)

if __name__ == '__main__':
    DATASET_CONFIGS = {
        'v1': {
            "handle": "andrewmvd/cancer-instance-segmentation-and-classification",
            "num_classes": 3,
            "class_map": {1: 'Neoplastic', 2: 'Inflammatory', 5: 'Epithelial'}
        }
        # Add configs for v2, v3 here if needed
    }
    parser = argparse.ArgumentParser(description="Train an Instance-Level (Multi-Class) Classifier.")
    parser.add_argument('--config', default='v1', choices=DATASET_CONFIGS.keys(), help="Which dataset config to use.")
    args = parser.parse_args()
    
    config = DATASET_CONFIGS[args.config]
    raw_data_path = kagglehub.dataset_download(config['handle'])
    cropped_data_dir = f"./cropped_{config['handle'].split('/')[1]}"

    create_cropped_dataset(raw_data_path, cropped_data_dir, config['class_map'])
    
    classifier = NucleusClassifier(model_path=None, num_classes=config['num_classes'])
    classifier.train(
        dataset_path=cropped_data_dir, output_model_path=f"instance_model_{args.config}.pth",
        epochs=15, batch_size=128, lr=0.001
    )