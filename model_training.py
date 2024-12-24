import os
from PIL import Image
from torch.utils.data import Dataset

class DTDTextureDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        """
        txt_file: path to train1.txt, val1.txt, or test1.txt
        root_dir: path to the 'images' folder (where 47 subfolders are located)
        transform: any transformations (augmentations, normalization, etc.)
        """
        self.transform = transform
        self.root_dir = root_dir

        # Read the list of file names (banded/banded_0001.jpg, ...)
        with open(txt_file, 'r') as f:
            lines = f.read().splitlines()

        # Example line: "banded/banded_0001.jpg"
        # The class is derived as the part of the path up to the first slash.
        self.samples = []
        self.classes_set = set()
        for line in lines:
            category = line.split('/')[0]  # "banded"
            self.samples.append((line, category))
            self.classes_set.add(category)

        # As a result, self.samples will contain [(rel_path, class_name), ...]

        # Create a list of all categories (47 items) in **sorted** order
        self.classes_list = sorted(list(self.classes_set))
        # Dictionary {class: index}, to return numbers (0..46)
        self.class_to_idx = {cat: i for i, cat in enumerate(self.classes_list)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rel_path, category = self.samples[index]
        img_path = os.path.join(self.root_dir, rel_path)  
        # Example: "dtd/images/banded/banded_0001.jpg"

        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[category]

        if self.transform:
            image = self.transform(image)

        return image, label

from torchvision import transforms

IMG_SIZE = 224

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])


from torch.utils.data import DataLoader

train_file = "dtd/labels/train1.txt"
val_file   = "dtd/labels/val1.txt"
root_dir   = "dtd/images"

train_dataset = DTDTextureDataset(train_file, root_dir, transform=train_transforms)
val_dataset   = DTDTextureDataset(val_file,   root_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

# quantity of the classes:
num_classes = len(train_dataset.classes_list)  # 47
print("Классы:", train_dataset.classes_list)


import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet50 on ImageNet
model = models.resnet50(pretrained=True)

# Freeze layers (optional — if we want to train only the last layer)
for param in model.parameters():
    param.requires_grad = False

# Replace the output layer with 47 classes
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)  # num_classes = 47

model = model.to(device)


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

num_epochs = 5
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"Epoch [{epoch+1}/{num_epochs}]: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
