import torch
from torchvision import models
from torch import nn
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "product_model.pth"

# -------------------------------------------------
# Model loader
# -------------------------------------------------
def load_model(num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load trained weights safely
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )

    model.eval()
    return model