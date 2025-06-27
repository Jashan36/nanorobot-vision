# run_pipeline.py
# Script to run the full pipeline on a single image.
import argparse
from nanorobot_vision import NanoVisionSystem

def main():
    parser = argparse.ArgumentParser(description='Nanorobot Vision Pipeline for a single image.')
    parser.add_argument('image_path', help='Path to input microscopy image')
    parser.add_argument('--output', '-o', default='output.jpg', help='Output visualization path')
    parser.add_argument('--seg-model', default='cyto', help='Cell segmentation model type')
    parser.add_argument('--cls-model', default='kather_model.pth', help='Tissue classifier model path')
    args = parser.parse_args()
    
    pipeline = NanoVisionSystem(seg_model=args.seg_model, cls_model_path=args.cls_model)
    results = pipeline.process(args.image_path, args.output)
    
    print("\n=== Analysis Results ===")
    print(f"  Detected cells: {results['segmentation']['cell_count']}")
    print(f"  Tissue type: {results['classification']['class']}")
    print(f"  Confidence: {results['classification']['confidence']:.4f}")
    print(f"  Visualization saved to: {args.output}")

if __name__ == "__main__":
    main()
