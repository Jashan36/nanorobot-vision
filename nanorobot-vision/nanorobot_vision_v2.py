# nanorobot_vision_v2.py
# Orchestrator for instance-level segmentation and classification.
import cv2
import numpy as np
from cell_segmenter import CellSegmenter
from nucleus_classifier import NucleusClassifier
from typing import Dict, Any, List
import pandas as pd
from PIL import Image
from scipy.ndimage import center_of_mass

class InstanceVisionSystem:
    def __init__(self, nuc_cls_model_path: str, num_classes: int, class_names: List[str]):
        """
        Initializes the vision system for instance-level analysis.
        :param nuc_cls_model_path: Path to the trained nucleus classifier model.
        :param num_classes: The number of classes the model was trained on.
        :param class_names: The names of the classes, in order.
        """
        self.segmenter = CellSegmenter(model_type='nuclei', gpu=True)
        self.classifier = NucleusClassifier(model_path=nuc_cls_model_path, num_classes=num_classes)
        self.classifier.class_names = class_names  # Manually set class names for inference
        self.crop_size = 48

    def _load_image(self, path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None: raise ValueError(f"Could not load image: {path}")
        # Convert BGR to RGB for PIL compatibility
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _extract_nucleus_crops(self, image: np.ndarray, masks: np.ndarray) -> List[Dict[str, Any]]:
        """Extracts PIL image crops for each nucleus instance."""
        nucleus_instances = []
        for label_id in np.unique(masks):
            if label_id == 0: continue
            
            cy, cx = center_of_mass(masks == label_id)
            cy, cx = int(cy), int(cx)
            
            y1, y2 = max(0, cy - self.crop_size // 2), min(image.shape[0], cy + self.crop_size // 2)
            x1, x2 = max(0, cx - self.crop_size // 2), min(image.shape[1], cx + self.crop_size // 2)
            
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                nucleus_instances.append({
                    'instance_id': label_id,
                    'crop': Image.fromarray(crop),
                    'centroid': (cx, cy)
                })
        return nucleus_instances

    def process_image(self, image_path: str) -> pd.DataFrame:
        """Processes a single image for instance-level classification."""
        image = self._load_image(image_path)
        seg_results = self.segmenter.segment(image)
        instances = self._extract_nucleus_crops(image, seg_results['masks'])
        
        crops_to_classify = [inst['crop'] for inst in instances]
        classification_results = self.classifier.classify_batch(crops_to_classify)
        
        results_list = []
        for i, instance in enumerate(instances):
            results_list.append({
                'image_name': os.path.basename(image_path),
                'instance_id': instance['instance_id'],
                'centroid_x': instance['centroid'][0],
                'centroid_y': instance['centroid'][1],
                'predicted_class': classification_results[i]['class'],
                'confidence': classification_results[i]['confidence']
            })
        return pd.DataFrame(results_list)
