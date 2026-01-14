# train_model.py

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import json

# -----------------------------
# 1️⃣ Device (CPU / GPU)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2️⃣ Image preprocessing
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 3️⃣ Load dataset (FIXED)
# -----------------------------
dataset = datasets.ImageFolder(
    root="datasets",
    transform=train_transform
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

# -----------------------------
# 4️⃣ Save class names (IMPORTANT)
# -----------------------------
with open("classes.json", "w") as f:
    json.dump(dataset.classes, f)

print("Classes:", dataset.classes)

# -----------------------------
# 5️⃣ Load pretrained ResNet18
# -----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(device)
model.train()

# -----------------------------
# 6️⃣ Training setup
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 7️⃣ Train model
# -----------------------------
EPOCHS = 15

for epoch in range(EPOCHS):
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# -----------------------------
# 8️⃣ Save model
# -----------------------------
torch.save(model.state_dict(), "product_model.pth")

print("✅ Training complete!")
print("✅ product_model.pth saved")
print("✅ classes.json saved")
