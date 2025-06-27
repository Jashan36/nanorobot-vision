# nucleus_classifier.py
# Module for a lightweight classifier designed for individual nucleus crops.
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, datasets, transforms
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

class NucleusClassifier:
    def __init__(self, model_path: str = None, num_classes: int = 3):
        """
        Initializes the Nucleus Classifier.
        :param model_path: Path to trained model weights. If None, initializes with pre-trained ImageNet weights.
        :param num_classes: The number of classes to predict (e.g., Neoplastic, Inflammatory, etc.).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.class_names = []  # Will be populated during training from the dataset folders

        # Use a lightweight MobileNetV3 for efficient instance-level classification
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.Hardswish(), # Better activation for MobileNet
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(128, self.num_classes),
        )
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded trained nucleus classifier weights from {model_path}")
        
        self.model.to(self.device)
        self.inference_transform = self._get_inference_transform()

    def _get_inference_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify_batch(self, pil_images: List[Any]) -> List[Dict[str, Any]]:
        """Classifies a batch of PIL Image objects."""
        if not pil_images:
            return []
        
        self.model.eval()
        tensor_batch = torch.stack([self.inference_transform(img) for img in pil_images]).to(self.device)
        
        results = []
        with torch.no_grad():
            outputs = self.model(tensor_batch)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)

            for i in range(len(preds)):
                class_idx = preds[i].item()
                class_name = self.class_names[class_idx] if self.class_names and class_idx < len(self.class_names) else f"Class_{class_idx}"
                results.append({'class': class_name, 'confidence': confidences[i].item()})
        return results
    
    def train(self, dataset_path: str, output_model_path: str, epochs: int, batch_size: int, lr: float):
        """Trains the model on a dataset of cropped nuclei images."""
        print("--- Starting Nucleus Classifier Training ---")
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        full_dataset = datasets.ImageFolder(dataset_path)
        
        if self.num_classes != len(full_dataset.classes):
            raise ValueError(f"Model configured for {self.num_classes} classes, but dataset has {len(full_dataset.classes)} classes.")

        # CRITICAL: Get class names from the dataset folder structure
        self.class_names = full_dataset.classes
        print(f"Training on classes: {self.class_names}")

        train_size = int(0.85 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

        # Apply transforms to the datasets correctly after splitting
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = self._get_inference_transform()

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        print(f"Training on {len(train_subset)} images, validating on {len(val_subset)} images.")
        
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        best_val_accuracy = 0.0
        for epoch in range(epochs):
            self.model.train()
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            self.model.eval()
            val_corrects = 0
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    val_corrects += torch.sum(preds == labels.data)
            
            val_accuracy = val_corrects.double() / len(val_subset)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), output_model_path)
                print(f"  -> New best model saved with accuracy: {best_val_accuracy:.4f}")
