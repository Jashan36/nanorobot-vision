# nanorobot_vision.py
# Main pipeline logic to integrate segmentation and classification.
import cv2
import numpy as np
from cell_segmenter import CellSegmenter
from tissue_classifier import TissueClassifier
from typing import Dict, Any, Optional

class NanoVisionSystem:
    def __init__(self, seg_model: str = 'cyto', cls_model_path: str = 'tissue_classifier.pth'):
        """
        Initialize the integrated vision system.
        :param seg_model: Cell segmentation model type.
        :param cls_model_path: Path to the trained tissue classifier model.
        """
        print("Initializing Cell Segmenter...")
        self.segmenter = CellSegmenter(model_type=seg_model)
        
        # Automatically determine number of classes for the classifier
        # This is a bit of a heuristic; assumes a standard state dict.
        num_classes = self._get_num_classes_from_model(cls_model_path)
        print(f"Initializing Tissue Classifier for {num_classes} classes...")
        self.classifier = TissueClassifier(model_path=cls_model_path, num_classes=num_classes)

    def _get_num_classes_from_model(self, model_path: str) -> int:
        """Introspects a model file to find the number of output classes."""
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            # Assuming the last layer is named 'fc.3.weight' or 'fc.4.weight' etc.
            # This works for the defined ResNet structure.
            last_layer_key = [k for k in state_dict if k.endswith('.weight') and 'fc' in k][-1]
            return state_dict[last_layer_key].shape[0]
        except Exception:
            print("Warning: Could not determine num_classes from model file. Defaulting to 9 for Kather dataset.")
            return 9 # Fallback

    def process(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an input image through the full segmentation and classification pipeline.
        :param image_path: Path to the input image.
        :param output_path: Optional path to save the annotated visualization.
        :return: A dictionary containing combined results.
        """
        image = self._load_image(image_path)
        
        seg_results = self.segmenter.segment(image)
        cls_results = self.classifier.classify(image)
        
        vis_image = self._create_visualization(image, seg_results, cls_results)
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return {
            'segmentation': seg_results,
            'classification': cls_results,
            'visualization': vis_image
        }

    def _load_image(self, path: str) -> np.ndarray:
        """Loads an image, raising an error if it fails."""
        # Using IMREAD_ANYCOLOR | IMREAD_ANYDEPTH handles various formats
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            # TODO: Add specialized microscopy format loading here (e.g., with OpenSlide for WSI)
            raise ValueError(f"Could not load image: {path}. Check path and file format.")
        return image

    def _create_visualization(self, image: np.ndarray, seg_results: Dict, cls_results: Dict) -> np.ndarray:
        """Creates a single annotated visualization image."""
        # Use the segmenter's visualizer as a base, which already draws contours and cell count
        vis = self.segmenter.visualize(image, seg_results)
        
        # Add classification information on top
        label = f"{cls_results['class']} ({cls_results['confidence']:.2f})"
        cv2.putText(vis, label, (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return vis
