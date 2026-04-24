"""
train_model.py
==============
Train a MobileNetV2-based emotion recognition CNN on the FER-2013 dataset.
Saves: emotion_model.pt  (PyTorch)
       emotion_model.onnx (for Unity Barracuda)

USAGE:
    python train_model.py --data_dir ./fer2013 --epochs 30 --batch_size 64
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Constants ────────────────────────────────────────────────────────────────
EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
NUM_CLASSES = len(EMOTIONS)
IMG_SIZE = 48
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"[INFO] Using device: {DEVICE}")


# ── Dataset ──────────────────────────────────────────────────────────────────
class FER2013Dataset(Dataset):
    """
    Expects FER-2013 folder structure:
        fer2013/
            train/
                Angry/  Disgusted/  Fearful/  Happy/  Neutral/  Sad/  Surprised/
            test/
                Angry/  Disgusted/ ...
    OR the original fer2013.csv file.
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.transform = transform
        self.samples = []

        # ── Option A: folder structure ───────────────────────────────────
        split_dir = os.path.join(root_dir, split)
        if os.path.isdir(split_dir):
            for idx, emotion in enumerate(EMOTIONS):
                emotion_dir = os.path.join(split_dir, emotion)
                if not os.path.isdir(emotion_dir):
                    continue
                for fname in os.listdir(emotion_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(emotion_dir, fname), idx))

        # ── Option B: CSV file ────────────────────────────────────────────
        else:
            csv_path = os.path.join(root_dir, 'fer2013.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Neither '{split_dir}' folder nor 'fer2013.csv' found in {root_dir}.\n"
                    "Download FER-2013 from: https://www.kaggle.com/datasets/msambare/fer2013"
                )
            df = pd.read_csv(csv_path)
            usage_map = {'train': 'Training', 'test': 'PublicTest'}
            df = df[df['Usage'] == usage_map.get(split, 'Training')]
            for _, row in df.iterrows():
                pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
                self.samples.append((pixels, int(row['emotion'])))
            self.csv_mode = True
            return
        self.csv_mode = False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item, label = self.samples[idx]
        if hasattr(self, 'csv_mode') and self.csv_mode:
            img = Image.fromarray(item).convert('RGB')
        else:
            img = Image.open(item).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, NUM_CLASSES)
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, 100. * correct / total, all_preds, all_labels


# ── Export to ONNX ────────────────────────────────────────────────────────────
def export_onnx(model, save_path):
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    torch.onnx.export(
        model, dummy, save_path,
        export_params=True, opset_version=11,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    print(f"[INFO] ONNX model saved → {save_path}")


# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_training(train_losses, val_losses, train_accs, val_accs, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses,   'r-', label='Val Loss')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()

    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs,   'r-', label='Val Acc')
    ax2.set_title('Accuracy (%)'); ax2.set_xlabel('Epoch'); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved.")


def plot_confusion(labels, preds, save_dir):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=EMOTIONS, yticklabels=EMOTIONS,
                cmap='Blues')
    plt.title('Confusion Matrix'); plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='./fer2013',  help='Path to FER-2013 dataset')
    parser.add_argument('--save_dir',   default='./models',   help='Where to save model files')
    parser.add_argument('--epochs',     type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Transforms ────────────────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ── Datasets & Loaders ────────────────────────────────────────────────
    print("[INFO] Loading dataset...")
    train_ds = FER2013Dataset(args.data_dir, split='train', transform=train_tf)
    val_ds   = FER2013Dataset(args.data_dir, split='test',  transform=val_tf)
    print(f"[INFO] Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Model, Loss, Optimizer ─────────────────────────────────────────────
    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training Loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, preds, labels = evaluate(model, val_loader, criterion)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs}] "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2f}% | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.2f}%")

        # Save best model
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            pt_path = os.path.join(args.save_dir, 'emotion_model.pt')
            torch.save({'model_state_dict': model.state_dict(),
                        'val_acc': vl_acc, 'emotions': EMOTIONS}, pt_path)
            print(f"  ✅ Best model saved → {pt_path} (Val Acc: {vl_acc:.2f}%)")

    # ── Final Evaluation ──────────────────────────────────────────────────
    print("\n[INFO] Final Evaluation Report:")
    print(classification_report(labels, preds, target_names=EMOTIONS, labels=list(range(len(EMOTIONS)))))
    plot_training(train_losses, val_losses, train_accs, val_accs, args.save_dir)
    plot_confusion(labels, preds, args.save_dir)

    # ── ONNX Export ───────────────────────────────────────────────────────
    onnx_path = os.path.join(args.save_dir, 'emotion_model.onnx')
    export_onnx(model, onnx_path)
    print(f"\n[DONE] Best Val Accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
