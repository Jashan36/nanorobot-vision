# cell_segmenter.py
# Module for cell segmentation logic
import cv2
import numpy as np
from cellpose import models, io
from skimage.measure import find_contours
from typing import Tuple, Dict, Any, List, Optional

class CellSegmenter:
    def __init__(self, model_type: str = 'cyto', gpu: bool = True):
        """
        Initialize Cellpose segmentation model
        :param model_type: Model type ('cyto' for cytoplasm, 'nuclei' for nuclei)
        :param gpu: Enable GPU acceleration
        """
        self.model = models.Cellpose(gpu=gpu, model_type=model_type)
        self.diameter = None  # Auto-detect diameter
        self.channels = [0, 0]  # Use [grayscale, grayscale]
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess microscopy image for Cellpose
        :param image: Input image (any format)
        :return: Preprocessed image (8-bit 3-channel)
        """
        # Handle different image types
        if image.dtype == np.uint16:
            image = (image / 65535 * 255).astype(np.uint8)
        elif image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        # Ensure 3 channels
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
            
        return image
    
    def segment(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform cell segmentation
        :param image: Input microscopy image
        :return: Dictionary containing masks, contours, and metadata
        """
        try:
            # Preprocess image
            proc_img = self.preprocess(image)
            
            # Run Cellpose model
            masks, flows, styles, diams = self.model.eval(
                proc_img,
                diameter=self.diameter,
                channels=self.channels,
                flow_threshold=0.4,
                cellprob_threshold=0.0
            )
            
            # Postprocess results
            contours = self._extract_contours(masks)
            cell_count = len(np.unique(masks)) - 1  # Exclude background
            
            return {
                'masks': masks,
                'contours': contours,
                'cell_count': cell_count,
                'diameter': diams,
                'flows': flows
            }
        except Exception as e:
            raise RuntimeError(f"Segmentation failed: {str(e)}")
    
    def _extract_contours(self, masks: np.ndarray) -> List[np.ndarray]:
        """
        Extract precise cell contours from masks
        :param masks: Segmentation mask array
        :return: List of contour points
        """
        contours = []
        for label in np.unique(masks):
            if label == 0:  # Skip background
                continue
            mask = (masks == label).astype(np.uint8)
            cnts = find_contours(mask, 0.5)
            for contour in cnts:
                # Convert from (row, col) to (x, y)
                contour = contour[:, ::-1].astype(np.int32)
                contours.append(contour)
        return contours
    
    def visualize(self, image: np.ndarray, results: Dict, save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize segmentation results
        :param image: Original input image
        :param results: Segmentation results dictionary
        :param save_path: Optional path to save visualization
        :return: Visualization image
        """
        # Convert to displayable format
        if len(image.shape) == 2:
            disp_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            disp_img = image.copy()
        
        if disp_img.dtype == np.uint16:
            disp_img = (disp_img / 65535 * 255).astype(np.uint8)
        
        # Draw all contours
        for contour in results['contours']:
            cv2.polylines(disp_img, [contour], True, (0, 255, 0), 1)
        
        # Add metadata
        cv2.putText(disp_img, f"Cells: {results['cell_count']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, disp_img)
        
        return disp_img
