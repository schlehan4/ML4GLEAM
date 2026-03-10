import torch
import torch.nn as nn

class RGZ_ResNet34Head(nn.Module):
    def __init__(self,base_model, in_features, num_classes, dropout_rate=0.5):
        super(RGZ_ResNet34Head, self).__init__()

        # ResNet-34 laden (ohne die letzte Fully-Connected-Schicht)
        self.base_model = base_model
        self.base = nn.Sequential(*list(self.base_model.children())[:-1])  # Entfernt die FC-Schicht

        # Deep Classification Head
        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.fc3 = nn.Linear(256, num_classes)  # Finale Klassifikation (keine Softmax!)

    def forward(self, x):
        x = self.base(x)  # Feature Extraction durch ResNet-34
        x = torch.flatten(x, 1)  # [B, 512, 1, 1] → [B, 512]

        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)  # Logits für CrossEntropyLoss
        return x
#
# # 🔹 ResNet-34 laden & neuen Head einfügen
# model = models.resnet34(pretrained=True)
# model = RGZ_ResNetHead(model,in_features=512, num_classes=6, dropout_rate=0.5)
#
# # 🔹 Falls du auf GPU trainierst:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# print(model)  # Kontrolle der Architektur
