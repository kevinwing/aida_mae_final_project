import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

class YoloDataset(Dataset):
    def __init__(self, root, target_size=(1024, 1024), transform=None):
        """
        Custom Dataset for loading YOLO-format labels.

        Args:
            root (str): Root directory containing 'images' and 'labels' folders.
            target_size (tuple): Resize target for images (width, height).
            transform: PyTorch transforms to apply to images.
        """
        self.image_dir = os.path.join(root, "images")
        self.label_dir = os.path.join(root, "labels")
        self.target_size = target_size
        self.transform = transform

        self.image_files = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_files)

    def parse_label_file(self, label_file, img_width, img_height):
        """
        Parse the YOLO label file to extract bounding boxes and class IDs.

        Args:
            label_file (str): Path to the YOLO label file.
            img_width (int): Width of the image.
            img_height (int): Height of the image.

        Returns:
            dict: A dictionary with 'boxes' and 'labels' tensors.
        """
        boxes = []
        labels = []
        with open(label_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(float(parts[0]))  # Convert class ID to integer
                x_center, y_center, width, height = map(float, parts[1:])

                # Convert normalized YOLO format to pixel coordinates
                xmin = (x_center - width / 2) * img_width
                ymin = (y_center - height / 2) * img_height
                xmax = (x_center + width / 2) * img_width
                ymax = (y_center + height / 2) * img_height

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

    def __getitem__(self, idx):
        # Load the image
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert("RGB")

        # Resize the image
        original_width, original_height = image.size
        image = image.resize(self.target_size)

        # Load the corresponding label file
        label_file = os.path.join(self.label_dir, os.path.splitext(image_file)[0] + ".txt")
        if os.path.exists(label_file):
            target = self.parse_label_file(label_file, original_width, original_height)
        else:
            # Handle empty label files
            target = {"boxes": torch.empty((0, 4), dtype=torch.float32), "labels": torch.empty((0,), dtype=torch.int64)}

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, target

def split_dataset(dataset, train_split=0.8, val_split=0.1):
    """
    Split dataset into train, validation, and test sets.

    Args:
        dataset: Instance of YoloDataset.
        train_split (float): Fraction of the dataset to use for training.
        val_split (float): Fraction of the training set to use for validation.

    Returns:
        tuple: Train, validation, and test datasets.
    """
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    test_size = total_size - train_size
    val_size = int(train_size * val_split)
    train_size -= val_size

    return random_split(dataset, [train_size, val_size, test_size])

# Example usage
if __name__ == "__main__":
    # Load the dataset
    dataset = YoloDataset(root="/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset", target_size=(1024, 1024))

    # Split into train, val, and test sets
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Print dataset sizes
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

