import tensorflow as tf
import os


def resize_with_padding(image, target_size):
    """
    Resize the image to the target size while maintaining the aspect ratio by adding padding.

    Args:
        image: TensorFlow image tensor.
        target_size: Tuple specifying the target height and width (e.g., (1024, 1024)).

    Returns:
        Resized image with padding.
        Scaling factors (scale_y, scale_x) for height and width.
        Padding offsets (offset_y, offset_x) for top and left padding.
    """
    target_height, target_width = target_size
    img_height, img_width = tf.shape(image)[0], tf.shape(image)[1]

    # Compute scaling factors
    scale = tf.minimum(target_width / tf.cast(img_width, tf.float32),
                       target_height / tf.cast(img_height, tf.float32))
    new_width = tf.cast(tf.round(tf.cast(img_width, tf.float32) * scale), tf.int32)
    new_height = tf.cast(tf.round(tf.cast(img_height, tf.float32) * scale), tf.int32)

    # Resize the image
    image_resized = tf.image.resize(image, [new_height, new_width])

    # Compute padding
    pad_height = target_height - new_height
    pad_width = target_width - new_width
    offset_y = pad_height // 2
    offset_x = pad_width // 2

    # Pad the image to the target size
    image_padded = tf.image.pad_to_bounding_box(image_resized, offset_y, offset_x, target_height, target_width)

    return image_padded, scale, offset_y, offset_x


def adjust_bboxes(bboxes, scale, offset_y, offset_x, img_width, img_height, target_size):
    """
    Adjust bounding boxes to account for resizing and padding.

    Args:
        bboxes: Tensor of bounding boxes in [xmin, ymin, xmax, ymax] format normalized to [0, 1].
        scale: Scaling factor used to resize the image.
        offset_y: Top padding offset.
        offset_x: Left padding offset.
        img_width: Original image width.
        img_height: Original image height.
        target_size: Tuple specifying the target height and width.

    Returns:
        Adjusted bounding boxes in normalized [xmin, ymin, xmax, ymax] format.
    """
    target_height, target_width = target_size

    # Scale bounding boxes to the resized image dimensions
    bboxes = bboxes * tf.constant([img_width, img_height, img_width, img_height], dtype=tf.float32)
    bboxes = bboxes * scale

    # Add padding offsets
    bboxes = bboxes + tf.constant([offset_x, offset_y, offset_x, offset_y], dtype=tf.float32)

    # Normalize to [0, 1] in the padded image
    bboxes = bboxes / tf.constant([target_width, target_height, target_width, target_height], dtype=tf.float32)

    return bboxes


def parse_image(image_path, target_size=(1024, 1024)):
    """
    Read, resize, and pad an image while maintaining its aspect ratio.

    Args:
        image_path: Path to the image file.
        target_size: Tuple specifying the target height and width (e.g., (1024, 1024)).

    Returns:
        Padded image.
        Scaling factors and padding offsets for bounding box adjustment.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)

    img_height, img_width = tf.shape(img)[0], tf.shape(img)[1]

    # Resize and pad the image
    img_padded, scale, offset_y, offset_x = resize_with_padding(img, target_size)

    return img_padded, img_width, img_height, scale, offset_y, offset_x


def parse_yolo_labels(label_path, img_width, img_height, scale, offset_y, offset_x, target_size=(1024, 1024)):
    """
    Parse YOLO labels, adjust for resizing and padding, and normalize to [0, 1].

    Args:
        label_path: Path to the YOLO `.txt` file.
        img_width: Original image width.
        img_height: Original image height.
        scale: Scaling factor used to resize the image.
        offset_y: Top padding offset.
        offset_x: Left padding offset.
        target_size: Tuple specifying the target height and width (e.g., (1024, 1024)).

    Returns:
        Adjusted bounding boxes in normalized [xmin, ymin, xmax, ymax] format.
        Class labels corresponding to the bounding boxes.
    """
    # Read YOLO labels from the file
    labels = tf.io.read_file(label_path)
    labels = tf.strings.strip(labels)  # Remove extra whitespace
    labels = tf.strings.split(labels, '\n')

    # Check for empty label file
    def is_valid_label(label):
        return tf.strings.length(label) > 0

    labels = tf.boolean_mask(labels, is_valid_label)

    # Return empty tensors if no labels are found
    if tf.size(labels) == 0:
        return tf.zeros((0, 4), dtype=tf.float32), tf.zeros((0,), dtype=tf.float32)

    # Parse each line into [class_id, center_x, center_y, width, height]
    parsed = tf.map_fn(
        lambda line: tf.strings.to_number(tf.strings.split(line), out_type=tf.float32),
        labels,
        fn_output_signature=tf.float32
    )

    # Ensure parsed tensor has the expected shape
    parsed = tf.ensure_shape(parsed, [None, 5])

    # Extract individual components
    class_ids = parsed[:, 0]
    center_x, center_y, bbox_width, bbox_height = tf.unstack(parsed[:, 1:], axis=1)

    # Convert YOLO format to [xmin, ymin, xmax, ymax]
    xmin = center_x - (bbox_width / 2)
    ymin = center_y - (bbox_height / 2)
    xmax = center_x + (bbox_width / 2)
    ymax = center_y + (bbox_height / 2)

    bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=1)

    # Adjust bounding boxes for resizing and padding
    bboxes = adjust_bboxes(bboxes, scale, offset_y, offset_x, img_width, img_height, target_size)

    return bboxes, class_ids


def parse_fn(image_path, label_path):
    target_size = 1024
    try:
        img, img_width, img_height, scale, offset_y, offset_x = parse_image(image_path, target_size)
        bboxes, class_ids = parse_yolo_labels(label_path, img_width, img_height, scale, offset_y, offset_x, target_size)

        # Skip if no valid labels are found
        if tf.size(class_ids) == 0:
            return img, {"bboxes": tf.zeros((0, 4), dtype=tf.float32), "class_ids": tf.zeros((0,), dtype=tf.float32)}

        return img, {"bboxes": bboxes, "class_ids": class_ids}

    except tf.errors.InvalidArgumentError as e:
        tf.print("Error parsing file:", image_path, label_path, e)
        return img, {"bboxes": tf.zeros((0, 4), dtype=tf.float32), "class_ids": tf.zeros((0,), dtype=tf.float32)}


def load_yolo_dataset(image_dir, label_dir, batch_size, target_size=(1024, 1024)):
    """
    Load the YOLO dataset and prepare it for Faster R-CNN training.

    Args:
        image_dir: Directory containing images.
        label_dir: Directory containing YOLO labels.
        batch_size: Batch size for the dataset.
        target_size: Tuple specifying the target image size (e.g., (1024, 1024)).

    Returns:
        A TensorFlow dataset.
    """
    image_paths = tf.data.Dataset.list_files(os.path.join(image_dir, "*.jpg"))
    label_paths = tf.data.Dataset.list_files(os.path.join(label_dir, "*.txt"))

    dataset = tf.data.Dataset.zip((image_paths, label_paths))

    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == '__main__':
    dataset = load_yolo_dataset(
        "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/images/val",
        "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/labels/val",
        batch_size=8
    )
    
    for img, labels in dataset.take(1):
        tf.print("Image Shape:", tf.shape(img))
        tf.print("Bounding Boxes:", labels["bboxes"], summarize=-1)
        tf.print("class IDs:", labels["class_ids"], summarize=-1)
