import tensorflow as tf
import os
import numpy as np

MAX_BOXES = 100


def yolo_to_tf(label_contents, width, height):
    """
    Convert YOLO bounding box format to TensorFlow bounding box format.
    
    Parameters:
    - label_contents: A TensorFlow string tensor containing YOLO format labels as text.
    - width: The image width (after resizing).
    - height: The image height (after resizing).
    
    Returns:
    - Tensor of shape (num_boxes, 5) containing [class_id, xmin, ymin, xmax, ymax].
    """
    # Split label contents into lines and parse numbers
    labels = tf.strings.split(label_contents, '\n')  # Split into lines
    labels = tf.strings.to_number(tf.strings.split(labels), tf.float32)  # Convert to floats

    # Reshape to (num_boxes, 5) if there are valid labels
    labels = tf.reshape(labels, (-1, 5))
    # tf.print(f"Reshaped labels (num_boxes, 5): {labels}")

    # Separate components
    class_id = labels[:, 0]
    x_center = labels[:, 1] * width
    y_center = labels[:, 2] * height
    box_width = labels[:, 3] * width
    box_height = labels[:, 4] * height

    # Calculate xmin, ymin, xmax, ymax
    xmin = x_center - (box_width / 2)
    ymin = y_center - (box_height / 2)
    xmax = x_center + (box_width / 2)
    ymax = y_center + (box_height / 2)


    # Stack the converted boxes back together
    tf_boxes = tf.stack([class_id, xmin, ymin, xmax, ymax], axis=1)
    # tf.print(f"Tensorflow labels: {tf_boxes}")

    return tf_boxes


# def yolo_to_tf(label_file, image_width, image_height):
#     """Convert YOLO normalized bounding boxes to TensorFlow absolute pixel format."""
#     tf_boxes = []
#     with open(label_file, "r") as f:
#         lines = f.readlines()
#         if not lines:  # Handle empty label files
#             return tf.constant([], shape=(0, 5), dtype=tf.float32)
#         for line in lines:
#             class_id, x_center, y_center, width, height = map(float, line.split())
#             xmin = (x_center - width / 2) * image_width
#             ymin = (y_center - height / 2) * image_height
#             xmax = (x_center + width / 2) * image_width
#             ymax = (y_center + height / 2) * image_height
#             tf_boxes.append([class_id, xmin, ymin, xmax, ymax])
#     return tf.constant(tf_boxes, dtype=tf.float32)


def parse_dataset(image_path, label_path):
    """Parse an image and its labels into TensorFlow-compatible format."""
    # Load and preprocess the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [1024, 1024]) / 255.0  # Normalize to [0, 1]

    # Set fixed size for bounding boxes (e.g., max 50 boxes per image)
    max_boxes = MAX_BOXES
    height, width = 1024, 1024  # Fixed dimensions due to resizing

    # Load labels
    label_contents = tf.io.read_file(label_path)
    is_empty = tf.equal(tf.strings.length(label_contents), 0)

    def handle_empty_labels():
        # Return empty class labels and bounding boxes
        empty_boxes = tf.zeros((max_boxes, 5), dtype=tf.float32)
        return empty_boxes

    def handle_non_empty_labels():
        # Parse and convert YOLO format labels
        tf_boxes = yolo_to_tf(label_contents, width, height)
        # tf.print("Parsed Tensorflow Boxes: ", tf_boxes)

        # Pad bounding boxes to fixed size
        padding = [[0, max_boxes - tf.shape(tf_boxes)[0]], [0, 0]]
        tf_boxes_padded = tf.pad(tf_boxes, padding, "CONSTANT")
        return tf_boxes_padded

    # Handle empty vs non-empty labels
    tf_boxes_padded = tf.cond(is_empty, handle_empty_labels, handle_non_empty_labels)

    # Separate class labels and bounding box coordinates
    class_labels = tf_boxes_padded[:, 0]  # First column: class IDs
    bbox_coordinates = tf_boxes_padded[:, 1:]  # Remaining columns: bounding boxes

    return img, (class_labels, bbox_coordinates)


# def parse_dataset(image_path, label_path):
#     """Parse an image and its labels into TensorFlow-compatible format."""
#     # Load and preprocess the image
#     img = tf.io.read_file(image_path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize(img, [1024, 1024]) / 255.0
#
#     # Set fixed size for bounding boxes (e.g., max 50 boxes per image)
#     max_boxes = 50
#     height, width = 1024, 1024  # Fixed dimensions due to resizing
#
#     # Load and convert labels
#     label_contents = tf.io.read_file(label_path)
#     # labels = tf.strings.to_number(tf.strings.split(label_contents), tf.float32)
#     # labels = tf.reshape(labels, (-1, 5))
#     # label_path = label_path.numpy().decode("utf-8")  # Convert label path to string
#     tf_boxes = yolo_to_tf(label_contents, width, height)
#
#     # Pad bounding boxes to fixed size
#     padding = [[0, max_boxes - tf.shape(tf_boxes)[0]], [0, 0]]
#     tf_boxes_padded = tf.pad(tf_boxes, padding, "CONSTANT")
#
#     class_labels = tf_boxes_padded[:, 0]
#     bbox_coordinates = tf_boxes_padded[:, 1]
#
#     return img, (class_labels, bbox_coordinates)


def _parse(image_path, label_path, normalize_bbox=False, model_type="faster_rcnn", max_rois=100):
    """
    Generalized parsing function for CNN, R-FCN, and Faster R-CNN.
    Load and preprocess the dataset, with batching.
    Args:
        image_path: Path to the input image.
        label_path: Path to the corresponding label.
        normalize_bbox: Whether to normalize bounding boxes to [0, 1].
        model_type: The model type ('cnn', 'rfcn', 'faster_rcnn').
        max_rois: Maximum number of RoIs (used for R-FCN if applicable).

    Returns:
        Tuple of inputs and labels, formatted for the specified model type.
    """

    img, (class_labels, bbox_coordinates) = parse_dataset(image_path, label_path)

    if normalize_bbox:
        height, width = 1024, 1024  # Match resized image dimensions
        bbox_coordinates = tf.convert_to_tensor(bbox_coordinates, dtype=tf.float32)
        bbox_coordinates = tf.stack([
            bbox_coordinates[..., 0] / width,  # xmin
            bbox_coordinates[..., 1] / height,  # ymin
            bbox_coordinates[..., 2] / width,  # xmax
            bbox_coordinates[..., 3] / height  # ymax
        ], axis=-1)

    # Handle CNN input/output structure
    if model_type == "cnn":
        return img, {'class_output': class_labels, 'bbox_output': bbox_coordinates}

    # Handle R-FCN input/output structure (fixed max_rois)
    elif model_type == "rfcn":
        num_bboxes = tf.shape(bbox_coordinates)[0]
        bbox_coordinates = bbox_coordinates[:max_rois]  # Truncate if > max_rois
        class_labels = class_labels[:max_rois]

        # Pad to max_rois
        padding_size = max_rois - num_bboxes
        bbox_coordinates = tf.pad(bbox_coordinates, [[0, padding_size], [0, 0]], constant_values=-1)
        class_labels = tf.pad(class_labels, [[0, padding_size]], constant_values=-1)

        return {'input_1': img, 'input_2': bbox_coordinates}, {
            'class_output': class_labels,
            'bbox_output': bbox_coordinates
        }

    # Handle Faster R-CNN input/output structure (variable RoIs)
    elif model_type == "faster_rcnn":
        feature_map_shape = (32, 32)
        rpn_objectness,  rpn_bbox = generate_rpn_labels(feature_map_shape, bbox_coordinates)

        return {'input_1': img, 'rois': bbox_coordinates}, {
            'rpn_objectness': rpn_objectness,
            'rpn_bbox': rpn_bbox,
            'roi_class': class_labels,
            'roi_bbox': bbox_coordinates
        }

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def load_dataset(image_dir, label_dir, batch_size, normalize_bbox=False, model_type="cnn", max_rois=100):
    """
    Load a dataset with dynamic input structure based on the model type.

    Args:
        image_dir: Directory containing images.
        label_dir: Directory containing corresponding labels.
        batch_size: Batch size for dataset.
        normalize_bbox: Whether to normalize bounding boxes to [0, 1].
        model_type: The model type ('cnn', 'rfcn', 'faster_rcnn').
        max_rois: Maximum number of RoIs (used for R-FCN if applicable).

    Returns:
        A TensorFlow Dataset object formatted for the specified model type.
    """
    image_paths = tf.data.Dataset.list_files(f"{image_dir}/*.jpg")
    label_paths = tf.data.Dataset.list_files(f"{label_dir}/*.txt")

    dataset = tf.data.Dataset.zip((image_paths, label_paths))
    dataset = dataset.map(
        lambda image_path, label_path: _parse(image_path, label_path, normalize_bbox, model_type, max_rois),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# def load_dataset(image_dir, label_dir, batch_size=16):
#     """
#     Load and preprocess the dataset, with batching.
#     """
#     image_files = tf.data.Dataset.list_files(f"{image_dir}/*.jpg")
#     label_files = tf.data.Dataset.list_files(f"{label_dir}/*.txt")
#
#     def _parse(image_path, label_path):
#         img, (class_labels, bbox_coordinates) = parse_dataset(image_path, label_path)
#         # img.set_shape([1024, 1024, 3])
#         # class_labels.set_shape([50])
#         # bbox_coordinates.set_shape([50, 4])  # Max boxes = 50, 5 values per box
#         return img, {'class_output': class_labels, 'bbox_output': bbox_coordinates}
#
#     dataset = tf.data.Dataset.zip((image_files, label_files))
#     dataset = dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
#     dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#
#     return dataset
#

# def load_dataset(image_dir, label_dir):
#     """Create a TensorFlow dataset for the given image and label directories."""
#     image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")])
#     label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".txt")])
#
#     dataset = tf.data.Dataset.from_tensor_slices((image_files, label_files))
#
#     # Wrapper function to call parse_dataset with tf.py_function, ensuring compatibility with TensorFlow's map operation
#     def _parse(image_path, label_path):
#         return tf.py_function(func=parse_dataset, inp=[image_path, label_path], Tout=(tf.float32, tf.float32))
#
#     return dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)


def generate_rpn_labels(feature_map_shape, bbox_coordinates, img_shape=(1024, 1024)):
    """
    Generate RPN objectness labels for a fixed grid size.

    Args:
        feature_map_shape: Shape of the RPN feature map (height, width).
        bbox_coordinates: Ground truth bounding boxes (normalized).
        img_shape: Shape of the original image (height, width).

    Returns:
        rpn_objectness: Binary labels for objectness (foreground/background), shape: (feature_height, feature_width, 1).
        rpn_bbox: Dummy regression targets for now, shape: (feature_height, feature_width, 4).
    """
    feature_height, feature_width = feature_map_shape
    img_height, img_width = img_shape

    # Map bbox coordinates to feature map space
    bbox_coordinates = bbox_coordinates * [feature_width, feature_height, feature_width, feature_height]

    # Initialize objectness labels with 0 (background)
    rpn_objectness = tf.zeros([feature_height, feature_width, 1], dtype=tf.float32)

    # Map each bounding box to the grid
    for bbox in bbox_coordinates:
        xmin, ymin, xmax, ymax = bbox
        # Compute the grid cell corresponding to the center of the bbox
        bbox_center = tf.cast([(xmin + xmax) / 2, (ymin + ymax) / 2], tf.int32)
        bbox_center_y, bbox_center_x = bbox_center[1], bbox_center[0]

        # Ensure indices are within bounds
        bbox_center_y = tf.clip_by_value(bbox_center_y, 0, feature_height - 1)
        bbox_center_x = tf.clip_by_value(bbox_center_x, 0, feature_width - 1)

        # Assign objectness = 1 for this grid cell
        rpn_objectness = tf.tensor_scatter_nd_update(
            rpn_objectness,
            indices=[[bbox_center_y, bbox_center_x, 0]],
            updates=[1.0]
        )

    # Placeholder for RPN bbox regression labels (not implemented yet)
    rpn_bbox = tf.zeros([feature_height, feature_width, 4], dtype=tf.float32)

    return rpn_objectness, rpn_bbox


def analyze_bboxes(dataset):
    counts = []
    for _, lbl_batch in dataset:
        for lbl in lbl_batch['bbox_output']:
            valid_boxes = tf.reduce_sum(tf.cast(tf.reduce_any(lbl != 0, axis=-1), tf.int32))
            counts.append(valid_boxes.numpy())
    return counts


if __name__ == "__main__":
    # image_dir = "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/images/train"
    # label_dir = "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/labels/train"

    # image_dir = "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/images/val"
    # label_dir = "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/labels/val"

    image_dir = "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/images/test"
    label_dir = "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/labels/test"

    # Example usage
    train_dataset = load_dataset(image_dir, label_dir, batch_size=8)
    bbox_counts = analyze_bboxes(train_dataset)
    print("Max bounding boxes in dataset:", max(bbox_counts))
    print("Bounding box distribution:", np.histogram(bbox_counts, bins=[0, 10, 20, 50, 100, 200]))

    for img_batch, lbl_batch in train_dataset.take(1):
        print(f"Images shape: {img_batch.shape}")
        print(f"Class labels shape: {lbl_batch['class_output'].shape}")
        print(f"BBox labels shape: {lbl_batch['bbox_output'].shape}")

