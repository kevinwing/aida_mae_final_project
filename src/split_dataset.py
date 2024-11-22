import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
image_dir = "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset/images"
label_dir = "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset/labels"
output_dir = "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split"

train_images = os.path.join(output_dir, "images/train")
val_images = os.path.join(output_dir, "images/val")
test_images = os.path.join(output_dir, "images/test")
train_labels = os.path.join(output_dir, "labels/train")
val_labels = os.path.join(output_dir, "labels/val")
test_labels = os.path.join(output_dir, "labels/test")

# Create directories
os.makedirs(train_images, exist_ok=True)
os.makedirs(val_images, exist_ok=True)
os.makedirs(test_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)
os.makedirs(test_labels, exist_ok=True)

# Get image and label paths
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")])
label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".txt")])

# Ensure matching image and label files
assert len(image_files) == len(label_files), "Number of images and labels do not match!"
image_files.sort()
label_files.sort()

# Split dataset: 80% train, 20% test
train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(image_files, label_files, test_size=0.2, random_state=42)

# Further split train set: 10% of train goes to val
train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(train_imgs, train_lbls, test_size=0.1, random_state=42)

# Copy files to respective directories
for imgs, lbls, img_dir, lbl_dir in [
    (train_imgs, train_lbls, train_images, train_labels),
    (val_imgs, val_lbls, val_images, val_labels),
    (test_imgs, test_lbls, test_images, test_labels),
]:
    for img, lbl in zip(imgs, lbls):
        shutil.copy(img, img_dir)
        shutil.copy(lbl, lbl_dir)

print("Dataset successfully split and organized!")

