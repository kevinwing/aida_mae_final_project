import os
from PIL import Image
import torch
from torchvision.transforms import functional as F

class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, target_size=(1024, 1024)):
        """
        Custom Dataset for loading YOLO-format labels.

        Args:
            root (str): Root directory containing 'images' and 'labels' folders.
            transforms: Optional transforms to apply to the images and targets.
            target_size (tuple): Target size to resize images (width, height).
        """
        self.root = root
        self.transforms = transforms
        self.target_size = target_size
        self.image_dir = os.path.join(root, "images")
        self.label_dir = os.path.join(root, "labels")
        self.image_files = sorted(os.listdir(self.image_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

    def __len__(self):
        return len(self.image_files)

    def parse_yolo_labels(self, label_file, img_width, img_height):
        """
        Parse YOLO label files and convert to absolute coordinates.

        Args:
            label_file (str): Path to the YOLO label file.
            img_width (int): Original image width.
            img_height (int): Original image height.

        Returns:
            dict: A dictionary with bounding boxes and labels.
        """
        boxes = []
        labels = []
        with open(label_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(float(parts[0]))  # Convert to integer
                x_center, y_center, width, height = map(float, parts[1:])

                # Convert YOLO normalized format to absolute coordinates
                xmin = (x_center - width / 2) * img_width
                ymin = (y_center - height / 2) * img_height
                xmax = (x_center + width / 2) * img_width
                ymax = (y_center + height / 2) * img_height

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)

        # Handle cases where no annotations exist
        if not boxes:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        return boxes, labels

    def resize_image_and_boxes(self, image, boxes):
        """
        Resize images and adjust bounding boxes.

        Args:
            image (PIL.Image): The input image.
            boxes (torch.Tensor): Bounding boxes in absolute coordinates.

        Returns:
            tuple: Resized image and adjusted bounding boxes.
        """
        orig_width, orig_height = image.size
        target_width, target_height = self.target_size

        # Resize the image
        image = F.resize(image, self.target_size)

        if boxes.numel() == 0:
            return image, boxes

        # Scale bounding boxes
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height
        boxes[:, [0, 2]] *= scale_x  # Scale x_min and x_max
        boxes[:, [1, 3]] *= scale_y  # Scale y_min and y_max

        return image, boxes

    def __getitem__(self, idx):
        # Load the image
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size

        # Load the corresponding label file
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found for image: {img_file}")
        
        boxes, labels = self.parse_yolo_labels(label_path, img_width, img_height)

        if boxes.numel() == 0:
            return None

        # Resize the image and boxes
        image, boxes = self.resize_image_and_boxes(image, boxes)

        # Calculate area
        # if boxes.numel() == 0:
        #     area = torch.empty((0,), dtype=torch.float32)
        # else:
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }

        # Apply optional transforms
        if self.transforms:
            image, target = self.transforms(image, target)

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

    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])


def collate_fn(batch):
    """
    Custom collate function to handle empty targets in a batch.
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return [], []
    images, targets = zip(*batch)
    return list(images), list(targets)


if __name__ == '__main__':
    dataset = YoloDataset(root="./datasets/dataset", target_size=(300, 300))
    print(f"Total samples: {len(dataset)}")

    for idx in range(5):  # Check the first 5 samples
        sample = dataset[idx]
        print(f"Sample {idx}: {sample}")