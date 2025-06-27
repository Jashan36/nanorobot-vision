# Nanorobot Vision

A modular, extensible pipeline for cell segmentation and tissue classification in microscopy images using deep learning (Cellpose, ResNet-50). Designed for research, prototyping, and educational use.

---

## 🚀 Features
- **Cell Segmentation**: Accurate cell/nuclei segmentation using Cellpose.
- **Tissue Classification**: Transfer learning with ResNet-50 for robust tissue type prediction.
- **Metadata Handling**: Saves class names and model info for reproducibility.
- **Easy CLI**: Run full pipeline or training with simple commands.
- **Unit Tests**: Basic tests for classifier logic and preprocessing.
- **Modular Code**: Easily extend or integrate into larger projects.

---

## Directory Structure
```
nanorobot-vision/
├── cell_segmenter.py
├── evaluate_segmentation.py
├── nanorobot_vision.py
├── nanorobot_vision_v2.py
├── nucleus_classifier.py
├── requirements.txt
├── run_batch_pipeline.py
├── run_instance_analyzer.py
├── run_pipeline.py
├── sample_microscopy_image.png  # (test image)
├── tests/
│   └── test_tissue_classifier.py
├── tissue_classifier.pth  # (trained model weights)
├── tissue_classifier.py
├── train_classifier.py
├── train_image_classifier.py
├── train_multi_label_classifier.py
```

---

## 🛠️ Setup

1. Create and activate a Python environment (Conda recommended):
   ```sh
   conda create -n nanorobot python=3.10
   conda activate nanorobot
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

---

## ⚡ Quickstart

### Run the Full Pipeline on a Single Image
```sh
python run_pipeline.py sample_microscopy_image.png --output output.jpg --seg-model cyto --cls-model tissue_classifier.pth
```
- `sample_microscopy_image.png`: Input image
- `output.jpg`: Output visualization
- `cyto`: Cellpose model type (`cyto` or `nuclei`)
- `tissue_classifier.pth`: Path to trained classifier weights

### Train the Tissue Classifier
```sh
python train_classifier.py /path/to/dataset --num-classes 9 --output-model tissue_classifier.pth --epochs 10 --batch-size 64 --lr 0.001
```
- `/path/to/dataset`: Root folder with subfolders for each class
- The best model and class names metadata will be saved as `tissue_classifier.pth` and `tissue_classifier.pth.meta.json`.

---

## 🧪 Testing
Unit tests are in the `tests/` directory. Run with:
```sh
python -m unittest discover tests
```
- Example: `test_tissue_classifier.py` checks preprocessing and classification output structure.

---

## 📁 Sample Data
- `sample_microscopy_image.png` is a placeholder. Replace with your own test image.
- For classifier training, use a dataset structured as:
  ```
  dataset/
    ├── ADI/
    ├── BACK/
    ├── ...
    └── TUM/
  ```

---

## 📝 Model Metadata
- When training, class names are saved as metadata in a `.meta.json` file alongside the model weights.
- When loading a model, the pipeline will automatically use these class names if available.

---

## 🛡️ Error Handling & Troubleshooting
- The code includes robust error handling for file I/O, model loading, and input validation.
- Informative warnings and errors are printed to help with debugging.
- If you see `ImportError: cv2` or `numpy`, ensure all dependencies are installed and your environment is activated.
- For CUDA errors, check your PyTorch and CUDA versions are compatible.

---

## 🤝 Contributing
- Fork the repo and submit pull requests for improvements or bug fixes.
- Add new tests in the `tests/` directory for new features.
- Open issues for questions, bugs, or feature requests.

---

## 📚 Citation
If you use this codebase in your research, please cite the original Cellpose and ResNet papers, and this repository if appropriate.

---

## ℹ️ Notes
- Make sure to use the correct CUDA version if running on GPU.
- For advanced visualization, consider using napari or matplotlib.
- The code is modular and ready for extension or integration into larger projects.
- For large datasets, adjust batch size and number of workers for optimal performance.

---

## 💡 Inspiration
- [Cellpose: a generalist algorithm for cellular segmentation](https://www.nature.com/articles/s41592-020-01018-x)
- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
