import albumentations as A
import cv2
import os
import hashlib
import numpy as np

# Define directories
image_dir = "/home/krw/projects/surf2024/datasets/train/images"
annotation_dir = "/home/krw/projects/surf2024/datasets/train/labels"
save_image_dir = "/run/media/krw/PortableSSD/surf2024/augmented_dataset/dataset/images"
save_annotation_dir = "/run/media/krw/PortableSSD/surf2024/augmented_dataset/dataset/labels"
os.makedirs(save_image_dir, exist_ok=True)
os.makedirs(save_annotation_dir, exist_ok=True)

# Set target number of images
target_num_images = 5000
current_images = os.listdir(image_dir)
image_hashes = set()  # To track unique images

# Define augmentation pipeline with 90-degree rotations
transform = A.Compose([
    A.RandomRotate90(p=1.0),  # Only 90°, 180°, or 270° rotations
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.Blur(blur_limit=3, p=0.3),
    A.CLAHE(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

def get_image_hash(image):
    # Compute a hash of the image for uniqueness check
    return hashlib.md5(cv2.imencode('.jpg', image)[1]).hexdigest()  # Updated to '.jpg'

def clip_bboxes(bboxes):
    # Clip bounding boxes to ensure all values are between 0 and 1
    clipped_bboxes = []
    for bbox in bboxes:
        x_center, y_center, width, height = bbox
        # Clip each coordinate to the [0, 1] range
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0, min(1 - x_center, x_center))
        height = np.clip(height, 0, min(1 - y_center, y_center))
        clipped_bboxes.append([x_center, y_center, width, height])
    return clipped_bboxes

# Process images for augmentation
# num_images = 5000
i = 0
# while i < (target_num_images - len(current_images)):
for i in range(target_num_images - len(current_images)):
    img_path = os.path.join(image_dir, current_images[i % len(current_images)])
    img = cv2.imread(img_path)

    # Display the file being processed
    print(f"Processing file: {img_path}")

    # Load corresponding annotation
    base_filename = os.path.splitext(current_images[i % len(current_images)])[0]
    annotation_path = os.path.join(annotation_dir, f"{base_filename}.txt")
    with open(annotation_path, "r") as f:
        boxes = []
        category_ids = []
        for line in f:
            label, x_center, y_center, width, height = map(float, line.strip().split())
            boxes.append([x_center, y_center, width, height])  # YOLO format
            category_ids.append(int(label))

    # Print original bounding boxes
    print(f"Original bounding boxes for {img_path}: {boxes}")

    # Apply augmentations and print augmented bounding boxes
    try:
        augmented = transform(image=img, bboxes=boxes, category_ids=category_ids)
        augmented_img = augmented["image"]
        augmented_bboxes = clip_bboxes(augmented["bboxes"])  # Clip bboxes after augmenting
        augmented_category_ids = augmented["category_ids"]

        # Print augmented bounding boxes
        print(f"Augmented bounding boxes for {img_path}: {augmented_bboxes}")
        # i += 1

    except Exception as e:
        print(f"Error processing file {img_path}: {e}")
        continue

    # Check for uniqueness
    img_hash = get_image_hash(augmented_img)
    if img_hash in image_hashes:
        continue  # Skip duplicate image
    image_hashes.add(img_hash)

    # Save unique augmented image
    output_image_path = os.path.join(save_image_dir, f"img_{i}.jpg")  # Save as .jpg
    cv2.imwrite(output_image_path, augmented_img)

    # Save augmented annotations
    output_annotation_path = os.path.join(save_annotation_dir, f"img_{i}.txt")
    with open(output_annotation_path, "w") as f:
        for bbox, label in zip(augmented_bboxes, augmented_category_ids):
            x_center, y_center, width, height = bbox
            f.write(f"{label} {x_center} {y_center} {width} {height}\n")

