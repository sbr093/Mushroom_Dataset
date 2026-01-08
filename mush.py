import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import random
import warnings

# Suppress warnings that might clutter the output
warnings.filterwarnings("ignore")

# =========================================================
# CONFIG (Updated Model, Image Size)
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Using a larger image size (448) is common for high-performance vision models
IMG_SIZE = 448
BATCH_SIZE = 16
EPOCHS = 25
LR = 3e-4
WARMUP_EPOCHS = 3

# IMPORTANT: Update this path to where your mushroom dataset is located!
DATA_DIR = r"D:\Python\Mushroom_Dataset" 
print("Using device:", DEVICE)

# =========================================================
# DATASET
# =========================================================
# The Dataset class remains robust and does not need updating
class MushroomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        cls_map = {
            # Ensure these folder names match your data directory
            "edible mushroom sporocarp": 0,
            "poisonous mushroom sporocarp": 1,
        }

        for name, label in cls_map.items():
            folder = os.path.join(root, name)
            if not os.path.isdir(folder):
                print(f"Warning: Folder not found: {folder}")
                continue
            for f in os.listdir(folder):
                path = os.path.join(folder, f)
                if os.path.isfile(path):
                    self.samples.append((path, label))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)

# =========================================================
# TRANSFORMS (Advanced Augmentations)
# =========================================================
# RandAugment is modern and highly effective.
# TrivialAugment might be used for newer PyTorch versions, but RandAugment is solid.
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandAugment(num_ops=2, magnitude=10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    # Standard ImageNet normalization values
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# =========================================================
# LOAD DATA
# =========================================================
dataset = MushroomDataset(DATA_DIR, transform=train_tf)

if len(dataset) == 0:
    print("FATAL ERROR: Dataset is empty. Check DATA_DIR path and folder structure.")
    exit()

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# Apply test transform only to the validation subset
val_ds.dataset.transform = test_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # num_workers added for speed
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print("Train:", len(train_ds), "| Val:", len(val_ds))

# =========================================================
# MIXUP (No changes needed, the implementation is correct)
# =========================================================
def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)

# =========================================================
# MODEL (Using a more recent, high-performance model)
# =========================================================
# Change model name here. 'convnext_small_in22k' is a good starting point for high accuracy.
# Alternatives: 'swinv2_base_window12_192_22k', 'convnext_v2_small'
MODEL_NAME = "convnext_small_in22k" 
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2)
model.to(DEVICE)

# Set optimizer and scheduler based on best practices
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Automatic Mixed Precision (AMP) for faster training
scaler = torch.cuda.amp.GradScaler() 

# =========================================================
# TRAINING LOOP
# =========================================================
best_acc = 0

print(f"Starting training on {MODEL_NAME} for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # Apply Mixup/CutMix randomly
        if random.random() < 0.5:
            imgs, la, lb, lam = mixup_data(imgs, labels)
        else:
            la = lb = labels
            lam = 1

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = mixup_criterion(outputs, la, lb, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()

    # ------------------ Validation ------------------
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            preds.extend(out.argmax(1).cpu().numpy())
            true.extend(labels.numpy())

    acc = accuracy_score(true, preds)
    print(f"\nVal accuracy: {acc * 100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"best_{MODEL_NAME}_final.pth")
        print(" ✓  Model improved and saved")

print("\nBEST ACCURACY:", best_acc * 100)

# =========================================================
# TTA INFERENCE (Test-Time Augmentation)
# =========================================================
# TTA function is ready to be used after training to boost the final prediction accuracy
def tta_predict(model, img):
    # img should be a single, pre-processed image tensor (C, H, W)
    img = img.unsqueeze(0)
    aug = [
        img,
        torch.flip(img, dims=[3]), # Horizontal flip
        torch.rot90(img, 1, [2,3]), # 90 degree rotation
        torch.rot90(img, 2, [2,3]), # 180 degree rotation
    ]
    outs = []
    with torch.no_grad():
        for a in aug:
            outs.append(model(a.to(DEVICE)))
    
    # Average the logits from all augmented views
    avg_output = torch.stack(outs).mean(dim=0)
    return avg_output.argmax(1).item() # Return the final predicted class index