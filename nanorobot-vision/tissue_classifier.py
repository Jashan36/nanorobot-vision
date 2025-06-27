# multi_label_classifier.py
# A specialized classifier for multi-label image classification tasks like the Human Protein Atlas.
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from typing import List, Tuple

class MultiLabelClassifier:
    HPA_CLASSES = [
        "Nucleoplasm", "Nuclear membrane", "Nucleoli", "Nucleoli fibrillar center", "Nuclear speckles",
        "Nuclear bodies", "Endoplasmic reticulum", "Golgi apparatus", "Intermediate filaments",
        "Actin filaments", "Focal adhesion sites", "Microtubules", "Microtubule ends", "Cytokinetic bridge",
        "Mitochondria", "Microtubule organizing center", "Centrosome", "Lipid droplets", "Plasma membrane",
        "Cell junctions", "Vesicles", "Cytosol", "Aggresome", "Cytoplasmic bodies", "Rods & rings"
    ]
    NUM_CLASSES = len(HPA_CLASSES)

    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MultiLabelClassifier using device: {self.device}")

        self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1) # ResNet34 is a good balance
        num_ftrs = self.model.fc.in_features
        
        # --- KEY ARCHITECTURAL CHANGE ---
        # The final layer outputs logits for EACH class independently.
        # We do NOT use Softmax in the model definition.
        self.model.fc = nn.Linear(num_ftrs, self.NUM_CLASSES)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, output_model_path: str, epochs: int, lr: float):
        print("--- Starting Multi-Label Classifier Training ---")
        
        # --- KEY LOSS FUNCTION CHANGE ---
        # BCEWithLogitsLoss is the correct choice for multi-label problems.
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

        best_f1_score = 0.0
        for epoch in range(epochs):
            self.model.train()
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                images, labels = images.to(self.device), labels.to(self.device).float()
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # --- Validation with a Multi-Label Metric (F1 Score) ---
            self.model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    images = images.to(self.device)
                    outputs = self.model(images)
                    
                    # Apply sigmoid and a threshold (e.g., 0.5) to get binary predictions
                    preds = torch.sigmoid(outputs) > 0.5
                    
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
            
            # Calculate F1 score
            from sklearn.metrics import f1_score
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_labels)
            
            # Use 'macro' average to treat each class equally
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            print(f"Validation Macro F1 Score: {f1:.4f}")
            
            scheduler.step(f1) # Step scheduler based on validation F1

            if f1 > best_f1_score:
                best_f1_score = f1
                torch.save(self.model.state_dict(), output_model_path)
                print(f"  -> New best model saved with F1 score: {best_f1_score:.4f}")