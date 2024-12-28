import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def parse_yolo_label_file(label_file, img_width, img_height):
    """
    Parse the YOLO label file to extract bounding boxes and class IDs.

    Args:
        label_file (str): Path to the YOLO label file.
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        list: A list of bounding boxes in the format [class_id, xmin, ymin, xmax, ymax].
    """
    boxes = []
    with open(label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            # Convert normalized YOLO format to pixel coordinates
            xmin = (x_center - width / 2) * img_width
            ymin = (y_center - height / 2) * img_height
            xmax = (x_center + width / 2) * img_width
            ymax = (y_center + height / 2) * img_height

            boxes.append((class_id, xmin, ymin, xmax, ymax))
    return boxes

def visualize_yolo_bounding_boxes(image_dir, label_dir, num_images=5):
    """
    Visualize YOLO bounding boxes on random images.

    Args:
        image_dir (str): Path to the directory containing images.
        label_dir (str): Path to the directory containing YOLO label files.
        num_images (int): Number of images to visualize.
    """
    # Get all image files
    image_files = os.listdir(image_dir)

    # Randomly select a subset of images
    random_images = random.sample(image_files, num_images)

    for image_file in random_images:
        # Load the image
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size

        # Load the corresponding label file
        label_file = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")
        if not os.path.exists(label_file):
            print(f"Label file not found for image: {image_file}")
            continue

        # Parse the YOLO label file
        boxes = parse_yolo_label_file(label_file, img_width, img_height)

        # Plot the image
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        ax.axis("off")

        # Draw bounding boxes
        for class_id, xmin, ymin, xmax, ymax in boxes:
            width = xmax - xmin
            height = ymax - ymin

            # Create a rectangle patch
            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=2, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)

            # Add class label
            ax.text(
                xmin, ymin - 5, f"Class: {class_id}",
                color="red", fontsize=12, backgroundcolor="white"
            )

        # Show the plot
        plt.show()

# Example usage
if __name__ == "__main__":
    # Define the dataset structure
    image_dir = "./dataset/images"
    label_dir = "./dataset/labels"

    # Visualize YOLO bounding boxes
    visualize_yolo_bounding_boxes(image_dir, label_dir, num_images=5)

