# train_image_classifier.py
# A dedicated CLI script to train the image-level TissueClassifier.
import argparse
import kagglehub
from pathlib import Path
from tissue_classifier import TissueClassifier # Import the original classifier

def main():
    parser = argparse.ArgumentParser(description="Train an Image-Level Tissue Classifier.")
    parser.add_argument('--dataset', default="kmader/colorectal-histology-mnist", help="Kaggle dataset handle.")
    parser.add_argument('--output-model', '-o', default='histology_mnist_model.pth', help='Path to save the trained model.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    args = parser.parse_args()

    # --- This dataset has 8 classes ---
    NUM_CLASSES = 8

    print(f"Downloading dataset '{args.dataset}'...")
    dataset_path = kagglehub.dataset_download(args.dataset)

    # The dataset is often nested in another directory. We need to find the correct root.
    # The directory we want to pass to ImageFolder is the one that contains 'TUM', 'ADI', etc.
    data_root = Path(dataset_path)
    # Check if 'CRC-VAL-HE-7K' or similar is the parent of the class folders
    if (data_root / "Kather_texture_2016_image_data_for_Projet_Accasc" / "Kather_texture_2016_image_data").exists():
         data_root = data_root / "Kather_texture_2016_image_data_for_Projet_Accasc" / "Kather_texture_2016_image_data"
    
    print(f"Using data root for training: {data_root}")

    # Initialize a new, untrained classifier model
    classifier = TissueClassifier(model_path=None, num_classes=NUM_CLASSES)
    
    # Start the training process
    classifier.train(
        dataset_path=str(data_root),
        output_model_path=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    print(f"\nTraining complete. Model saved to '{args.output_model}'")

if __name__ == '__main__':
    main()
