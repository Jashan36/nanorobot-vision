# evaluate_segmentation.py
# A script to evaluate the segmentation performance of the system against a dataset with ground-truth masks.
import argparse
import kagglehub
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from nanorobot_vision_v2 import InstanceVisionSystem # We use this to get access to the segmenter
import cv2

def dice_coefficient(y_true, y_pred):
    """
    Calculates the Dice Coefficient for a single pair of binary masks.
    y_true: Ground truth mask (boolean or 0/1)
    y_pred: Predicted mask (boolean or 0/1)
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-9) # Add epsilon for stability

def find_corresponding_mask(image_path: Path, mask_dir: Path) -> Path:
    """Finds the ground truth mask file for a given image file."""
    # MoNuSeg naming convention, e.g., 'TCGA-18-5592-01Z-00-DX1.png' has mask 'TCGA-18-5592-01Z-00-DX1.xml'
    # The loader I'm assuming for simplicity gives a png mask.
    mask_path = mask_dir / image_path.name
    if not mask_path.exists():
        # Handle cases where mask has a different extension if needed
        pass
    return mask_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate segmentation performance.")
    parser.add_argument('--dataset', default="tuanledinh/monuseg2018", help="Kaggle dataset handle.")
    args = parser.parse_args()

    print(f"Downloading dataset '{args.dataset}'...")
    raw_data_path = Path(kagglehub.dataset_download(args.dataset))

    # --- Setup the System (we only need its segmenter) ---
    # We pass dummy values for the classifier part as it won't be used.
    print("Initializing Segmentation System...")
    system = InstanceVisionSystem(nuc_cls_model_path="dummy.pth", num_classes=1, class_names=["dummy"])

    # Define paths based on a common MoNuSeg structure
    # This might need adjustment based on the exact structure from Kagglehub
    train_images_dir = raw_data_path / 'MoNuSeg 2018 Training Data' / 'Tissue Images'
    train_masks_dir = raw_data_path / 'MoNuSeg 2018 Training Data' / 'Binary_masks_tiff_format' # Adjust if needed

    image_paths = list(train_images_dir.glob('*.tif'))
    if not image_paths:
        image_paths = list(train_images_dir.glob('*.png')) # Fallback
        
    if not image_paths:
        raise FileNotFoundError(f"Could not find images in {train_images_dir}")

    all_scores = []

    print(f"Evaluating segmentation on {len(image_paths)} images...")
    for img_path in tqdm(image_paths, desc="Evaluating Segmentation"):
        try:
            # 1. Get the model's prediction
            image = cv2.imread(str(img_path))
            pred_results = system.segmenter.segment(image)
            predicted_mask = pred_results['masks'] > 0  # Binarize: Is it a nucleus or not?

            # 2. Load the corresponding ground truth mask
            mask_filename = img_path.stem + "_mask.png"  # Adjust this if naming differs
            true_mask_path = train_masks_dir / mask_filename
            
            if not true_mask_path.exists():
                tqdm.write(f"Warning: Ground truth mask not found for {img_path.name}. Skipping.")
                continue
                
            ground_truth_mask = cv2.imread(str(true_mask_path), cv2.IMREAD_GRAYSCALE)
            ground_truth_mask = ground_truth_mask > 0 # Binarize

            # 3. Calculate the Dice score
            score = dice_coefficient(ground_truth_mask, predicted_mask)
            all_scores.append({'image': img_path.name, 'dice_score': score})

        except Exception as e:
            tqdm.write(f"[ERROR] Could not process {img_path.name}: {e}")

    # --- Generate and Print the Final Report ---
    if not all_scores:
        print("\nNo images were successfully evaluated.")
    else:
        results_df = pd.DataFrame(all_scores)
        mean_dice = results_df['dice_score'].mean()
        std_dice = results_df['dice_score'].std()
        
        print("\n" + "="*50)
        print("          SEGMENTATION PERFORMANCE REPORT")
        print("="*50)
        print(f"  Dataset: MoNuSeg 2018")
        print(f"  Segmentation Model: Cellpose ('nuclei')")
        print(f"  Images Evaluated: {len(results_df)}")
        print("-" * 50)
        print(f"  Mean Dice Score:   {mean_dice:.4f}")
        print(f"  Std Dev of Score:  {std_dice:.4f}")
        print("="*50)
        print("(Dice Score: 1.0 = perfect match, 0.0 = no overlap)")
        
        # Save detailed report
        results_df.to_csv("monuseg_evaluation_report.csv", index=False)
        print("\nDetailed report saved to 'monuseg_evaluation_report.csv'")