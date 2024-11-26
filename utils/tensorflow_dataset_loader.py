import tensorflow as tf
import os


def yolo_to_tf(label_file, image_width, image_height):
    """Convert YOLO normalized bounding boxes to TensorFlow absolute pixel format."""
    tf_boxes = []
    with open(label_file, "r") as f:
        lines = f.readlines()
        if not lines:  # Handle empty label files
            return tf.constant([], shape=(0, 5), dtype=tf.float32)
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.split())
            xmin = (x_center - width / 2) * image_width
            ymin = (y_center - height / 2) * image_height
            xmax = (x_center + width / 2) * image_width
            ymax = (y_center + height / 2) * image_height
            tf_boxes.append([class_id, xmin, ymin, xmax, ymax])
    return tf.constant(tf_boxes, dtype=tf.float32)


def parse_dataset(image_path, label_path):
    """Parse an image and its labels into TensorFlow-compatible format."""
    # Load and preprocess the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [1024, 1024]) / 255.0

    # Set fixed size for bounding boxes (e.g., max 50 boxes per image)
    max_boxes = 50
    height, width = 1024, 1024  # Fixed dimensions due to resizing

    # Load and convert labels
    label_path = label_path.numpy().decode("utf-8")  # Convert label path to string
    tf_boxes = yolo_to_tf(label_path, width, height)

    # Pad bounding boxes to fixed size
    padding = [[0, max_boxes - tf.shape(tf_boxes)[0]], [0, 0]]
    tf_boxes_padded = tf.pad(tf_boxes, padding, "CONSTANT")

    return img, tf_boxes_padded


def load_dataset(image_dir, label_dir):
    """Create a TensorFlow dataset for the given image and label directories."""
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".txt")])

    dataset = tf.data.Dataset.from_tensor_slices((image_files, label_files))

    # Wrapper function to call parse_dataset with tf.py_function, ensuring compatibility with TensorFlow's map operation
    def _parse(image_path, label_path):
        return tf.py_function(func=parse_dataset, inp=[image_path, label_path], Tout=(tf.float32, tf.float32))
    
    return dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

if __name__ == "__main__":
    image_dir = "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/images/train"
    label_dir = "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/labels/train"
    # Example usage
    train_dataset = load_dataset(image_dir, label_dir)
    train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    for batch in train_dataset.take(1):
        images, labels = batch
        print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")

