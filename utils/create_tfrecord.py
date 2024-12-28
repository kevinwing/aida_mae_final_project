import os
import tensorflow as tf
from object_detection.utils import dataset_util
from PIL import Image

# Paths
DATASET_DIR = "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/tensorflow_dataset"  # Update this with the dataset path
SPLITS = ["train", "val", "test"]  # Dataset splits
OUTPUT_DIR = os.path.join(DATASET_DIR, 'records')  # Directory to save TFRecords
LABEL_MAP = {"weed": 1}  # Update with your class labels

# Target dimensions
TARGET_SIZE = (1024, 1024)  # Desired size for all images (width, height)


def resize_and_pad(image, target_size=(1024, 1024), pad_color=(0, 0, 0)):
    """
    Resize an image while maintaining aspect ratio and pad to target size.

    Args:
        image (PIL.Image.Image): Input image.
        target_size (tuple): Target dimensions (width, height).
        pad_color (tuple): Color for padding (default: black).

    Returns:
        padded_image: The resized and padded image.
        new_width: New width of the resized image.
        new_height: New height of the resized image.
        original_width: Original width of the image.
        original_height: Original height of the image.
    """
    target_width, target_height = target_size
    original_width, original_height = image.size

    # Determine scale to maintain aspect ratio
    if original_width > original_height:  # Landscape
        scale = target_width / original_width
    else:  # Portrait
        scale = target_height / original_height

    # Resize image
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create new blank image with padding
    padded_image = Image.new("RGB", target_size, pad_color)
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    padded_image.paste(resized_image, (offset_x, offset_y))

    return padded_image, new_width, new_height, original_width, original_height, offset_x, offset_y


def create_tf_example(image_path, annotation_path, target_size):
    """Create a single TFRecord example with resized and padded image."""
    # Open and resize the image
    image = Image.open(image_path)
    padded_image, new_width, new_height, original_width, original_height, offset_x, offset_y = resize_and_pad(
        image, target_size
    )

    # Save resized image in memory for TFRecord
    import io
    buffer = io.BytesIO()
    padded_image.save(buffer, format="JPEG")
    encoded_image_data = buffer.getvalue()

    width, height = target_size  # After padding, dimensions are always 1024x1024
    filename = os.path.basename(image_path).encode('utf-8')
    image_format = b'jpg'  # Change if using PNG

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    # Read the bounding box annotations
    with open(annotation_path, 'r') as file:
        for line in file.readlines():
            class_id, xmin, ymin, xmax, ymax = map(float, line.strip().split())

            # Adjust bounding box coordinates to match resized image
            xmin = (xmin * new_width + offset_x) / width
            xmax = (xmax * new_width + offset_x) / width
            ymin = (ymin * new_height + offset_y) / height
            ymax = (ymax * new_height + offset_y) / height

            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)
            classes_text.append("weed".encode('utf-8'))  # Replace with actual class name
            classes.append(int(class_id))

    # Create a TF Example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_record(split_name, target_size, output_dir):
    """Create TFRecord files for a given dataset split."""
    images_dir = os.path.join(DATASET_DIR, split_name, "images")
    labels_dir = os.path.join(DATASET_DIR, split_name, "labels")
    output_file = os.path.join(output_dir, f"{split_name}.record")

    writer = tf.io.TFRecordWriter(output_file)
    for image_file in os.listdir(images_dir):
        print(f"Creating entry for {image_file}")
        if image_file.endswith('.jpg'):  # Adjust if needed
            annotation_file = os.path.join(labels_dir, os.path.splitext(image_file)[0] + ".txt")
            if os.path.exists(annotation_file):
                tf_example = create_tf_example(
                    os.path.join(images_dir, image_file),
                    annotation_file,
                    target_size
                )
                writer.write(tf_example.SerializeToString())
            else:
                print(f"Annotation file missing for {image_file}")
    writer.close()
    print(f"{split_name.capitalize()} TFRecord created at {output_file}")


# Main function to process all splits
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for split in SPLITS:
    create_tf_record(split, TARGET_SIZE, OUTPUT_DIR)

