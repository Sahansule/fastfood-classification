import os
import numpy as np
from torchvision import transforms, datasets
from sklearn.utils.class_weight import compute_class_weight

# Image dimensions
img_height = 224
img_width = 224

# Directory paths
base_dir = 'Fast Food Classification V2'
train_dir = os.path.join(base_dir, 'Train')
valid_dir = os.path.join(base_dir, 'Valid')
test_dir = os.path.join(base_dir, 'Test')

# ---------------------------------
# AUGMENTATION TRANSFORMS
# ---------------------------------

# Aggressive transforms for SimpleCNN
simple_cnn_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomResizedCrop(img_height, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Balanced transforms for DenseNet
densenet_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomResizedCrop(img_height, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),

    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Validation/Test transforms (same for both models)
valid_test_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------------------------------
# DATASET DEFINITIONS
# ---------------------------------

simple_cnn_train_dataset = datasets.ImageFolder(train_dir, transform=simple_cnn_transforms)
densenet_train_dataset    = datasets.ImageFolder(train_dir, transform=densenet_transforms)

valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
test_dataset  = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

# ---------------------------------
# CLASS WEIGHTS (shared)
# ---------------------------------

y_train_labels = np.array(simple_cnn_train_dataset.targets)
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights = {i: w for i, w in enumerate(class_weights_array)}

# ---------------------------------
# Print summary only when this script is run directly
# ---------------------------------

if __name__ == "__main__":
    print("\n📊 Training Dataset Info:")
    print(f" - Total training images: {len(simple_cnn_train_dataset)}")
    print(f" - Classes: {simple_cnn_train_dataset.classes}")
    for cls_idx, cls_name in enumerate(simple_cnn_train_dataset.classes):
        count = y_train_labels.tolist().count(cls_idx)
        print(f"   • {cls_name:<10} → {count} images")
