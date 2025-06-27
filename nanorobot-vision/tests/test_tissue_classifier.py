import unittest
import numpy as np
from tissue_classifier import TissueClassifier

class TestTissueClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = TissueClassifier(model_path=None, num_classes=9)

    def test_preprocess_shape(self):
        # Create a dummy image (BGR, 224x224)
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        tensor = self.classifier.preprocess(img)
        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))

    def test_classify_dummy(self):
        # Should not raise, but output is random since model is untrained
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = self.classifier.classify(img)
        self.assertIn('class', result)
        self.assertIn('confidence', result)
        self.assertIn('probabilities', result)

if __name__ == '__main__':
    unittest.main()
