import torch.nn as nn
import torch

class RGZ_SmallHead(nn.Module):
    def __init__(self, base_model, in_features, num_classes, dropout_rate=0.3):
        super(RGZ_SmallHead, self).__init__()

        self.base_model = base_model
        self.base = self.base_model.extract_features  # EfficientNet feature extractor
        self.pool = nn.AdaptiveAvgPool2d(1)           # Output: [B, 1280, 1, 1] → [B, 1280]

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.base(x)      # EfficientNet feature extraction
        x = self.pool(x)      # Adaptive pooling to [B, 1280, 1, 1]
        x = torch.flatten(x, 1)  # [B, 1280]
        x = self.classifier(x)
        return x

"""import torch
from efficientnet_pytorch import EfficientNet

# EfficientNet-B1 vorladen
model = EfficientNet.from_pretrained("efficientnet-b0")

# Beispiel-Eingabebild (Größe 3x224x224 für EfficientNet)
dummy_input = torch.randn(1, 3, 224, 224)

# Weitergabe der Eingabe durch das Modell, um die Form des Outputs zu prüfen
with torch.no_grad():  # Um keine Gradienten zu berechnen
    output = model.extract_features(dummy_input)  # Extrahiere Features ohne Klassifikationsschicht

print(output.shape)  # Ausgabe der Form der Feature-Map
"""