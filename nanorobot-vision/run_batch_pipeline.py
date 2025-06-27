# run_batch_pipeline.py
# Script to efficiently run the pipeline on a directory of images.
import argparse
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from nanorobot_vision import NanoVisionSystem

def create_summary_report_block(df: pd.DataFrame):
    """Prints a clear, separated block of summary statistics."""
    if df.empty:
        return "No results to summarize."

    total_images = len(df)
    class_distribution = df['tissue_type'].value_counts()
    avg_confidence = df['confidence'].mean()
    avg_cell_count = df['cell_count'].mean()

    report = []
    report.append("="*50)
    report.append("              BATCH ANALYSIS SUMMARY REPORT")
    report.append("="*50)
    report.append(f"  Total Images Processed: {total_images}")
    report.append(f"  Average Cell Count:     {avg_cell_count:.2f}")
    report.append(f"  Average Confidence:     {avg_confidence:.4f}")
    report.append("\n  --- Tissue Class Distribution ---")
    for cls, count in class_distribution.items():
        percentage = (count / total_images) * 100
        report.append(f"    {cls:<10s}: {count:<5d} images ({percentage:.2f}%)")
    report.append("="*50)
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Nanorobot Vision Batch Pipeline.')
    parser.add_argument('input_dir', help='Path to the directory of images.')
    parser.add_argument('--output-dir', '-o', default='batch_output', help='Directory to save results.')
    parser.add_argument('--cls-model', default='kather_model.pth', help='Path to the trained classifier model.')
    args = parser.parse_args()

    # --- Load models ONCE ---
    pipeline = NanoVisionSystem(cls_model_path=args.cls_model)

    # --- Setup paths and find images ---
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_vis_path = output_path / 'visualizations'
    output_vis_path.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(input_path.rglob('*.[pP][nN][gG]')) + \
                  list(input_path.rglob('*.[jJ][pP][gG]')) + \
                  list(input_path.rglob('*.[jJ][pP][eE][gG]')) + \
                  list(input_path.rglob('*.[tT][iI][fF]'))

    if not image_paths:
        print(f"No images found in '{args.input_dir}'. Exiting.")
        return

    # --- Loop, process, and collect results ---
    all_results = []
    print(f"\nProcessing {len(image_paths)} images...")
    for img_path in tqdm(image_paths, desc="Batch Analysis"):
        try:
            vis_save_path = output_vis_path / f"{img_path.stem}_result.jpg"
            results = pipeline.process(str(img_path), str(vis_save_path))
            all_results.append({
                'filename': img_path.name,
                'cell_count': results['segmentation']['cell_count'],
                'tissue_type': results['classification']['class'],
                'confidence': results['classification']['confidence']
            })
        except Exception as e:
            tqdm.write(f"  [ERROR] Skipping {img_path.name}: {e}")
            
    # --- Save CSV and print summary report block ---
    if all_results:
        results_df = pd.DataFrame(all_results)
        summary_csv_path = output_path / "summary_results.csv"
        results_df.to_csv(summary_csv_path, index=False)
        
        print("\nBatch processing complete.")
        print(f"Visualizations saved in: {output_vis_path}")
        print(f"Full CSV report saved to: {summary_csv_path}\n")

        # Create and print the clearly separated summary report block
        summary_block = create_summary_report_block(results_df)
        print(summary_block)
    else:
        print("No images were successfully processed.")

if __name__ == "__main__":
    main()
