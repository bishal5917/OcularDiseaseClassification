import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2

# Classes and Mapping
CLASS_NAMES = ['amd', 'cataract', 'diabetes', 'normal']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Dataset
class OcularClassificationDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(base_dir, cls_name)
            for img_path in glob.glob(os.path.join(cls_dir, '*.*')):
                self.image_paths.append(img_path)
                self.labels.append(CLASS_TO_IDX[cls_name])

        # print(self.image_paths)
        # print(self.labels)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# Transforms
train_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Datasets & Dataloaders
BATCH_SIZE = 8

train_dataset = OcularClassificationDataset(base_dir='Dataset/train', transform=train_transform)
val_dataset = OcularClassificationDataset(base_dir='Dataset/valid', transform=val_transform)
test_dataset = OcularClassificationDataset(base_dir='Dataset/test', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

