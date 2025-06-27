# run_instance_analyzer.py
import argparse
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from nanorobot_vision_v2 import InstanceVisionSystem
import json

def generate_summary_report(full_df: pd.DataFrame):
    """Creates a text block summarizing the results from all processed images."""
    if full_df.empty: return "No nuclei were processed."
    
    num_images = full_df['image_name'].nunique()
    total_nuclei = len(full_df)
    class_distribution = full_df['predicted_class'].value_counts(normalize=True) * 100
    
    report = ["="*50, "           BATCH INSTANCE ANALYSIS REPORT", "="*50]
    report.append(f"  Images Processed: {num_images}")
    report.append(f"  Total Nuclei Analyzed: {total_nuclei}")
    report.append(f"  Avg Nuclei per Image: {total_nuclei/num_images:.2f}\n")
    report.append("  --- Overall Nucleus Distribution ---")
    for cls, percentage in class_distribution.items():
        report.append(f"    {cls:<15s}: {percentage:.2f}%")
    report.append("="*50)
    return "\n".join(report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run instance-level analysis on a directory of images.")
    parser.add_argument('input_dir', help="Path to the directory of images to analyze.")
    parser.add_argument('--model-path', '-m', required=True, help="Path to the trained nucleus classifier model (.pth).")
    parser.add_argument('--output-csv', '-o', default='instance_analysis_results.csv', help="Path to save the final CSV report.")
    # You MUST provide the correct class info for the model you're using.
    parser.add_argument('--num-classes', type=int, required=True, help="Number of classes the model was trained on.")
    parser.add_argument('--class-names', nargs='+', required=True, help="Ordered list of class names (e.g., Benign Malignant).")
    args = parser.parse_args()

    # --- Initialize the system with the correct model config ---
    system = InstanceVisionSystem(
        nuc_cls_model_path=args.model_path,
        num_classes=args.num_classes,
        class_names=args.class_names
    )
    
    image_paths = list(Path(args.input_dir).rglob('*.[pP][nN][gG]')) # Add more extensions if needed
    if not image_paths:
        print(f"No .png images found in '{args.input_dir}'.")
        exit()

    # --- Process all images and concatenate results ---
    all_results = []
    print(f"Analyzing {len(image_paths)} images...")
    for img_path in tqdm(image_paths, desc="Analyzing Images"):
        try:
            df = system.process_image(str(img_path))
            all_results.append(df)
        except Exception as e:
            tqdm.write(f"[ERROR] Could not process {img_path.name}: {e}")
            
    if not all_results:
        print("No images were successfully processed.")
        exit()

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(args.output_csv, index=False)
    
    print(f"\nAnalysis complete. Full report saved to '{args.output_csv}'.\n")
    
    # Print summary report
    summary = generate_summary_report(final_df)
    print(summary)
