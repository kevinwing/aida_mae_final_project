import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def preprocess(image_path, label_path):
    """
    Preprocess a single image and its corresponding labels.
    
    Args:
        image_path (tf.Tensor): Path to the image file.
        label_path (tf.Tensor): Path to the label file.
    
    Returns:
        tf.Tensor, tf.Tensor: Preprocessed image and labels.
    """
    # Load and decode the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Decode image

    # Get original dimensions
    original_shape = tf.shape(img)[:2]  # [height, width]
    height, width = tf.cast(original_shape[0], tf.float32), tf.cast(original_shape[1], tf.float32)

    # Calculate scale factor to resize while preserving aspect ratio
    scale = 1024.0 / tf.maximum(height, width)
    new_height = tf.cast(height * scale, tf.int32)
    new_width = tf.cast(width * scale, tf.int32)

    # Resize the image while maintaining aspect ratio
    img = tf.image.resize(img, [new_height, new_width])

    # Pad the image to make it 1024 × 1024
    img = tf.image.resize_with_pad(img, 1024, 1024)

    # Normalize the image
    img = img / 255.0  # Normalize to [0, 1]

    # Load and parse labels
    def parse_labels(label_file):
        boxes = []
        with open(label_file.numpy().decode("utf-8"), "r") as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                # Convert YOLO format to TensorFlow format
                xmin = (x_center - width / 2) * width
                ymin = (y_center - height / 2) * height
                xmax = (x_center + width / 2) * width
                ymax = (y_center + height / 2) * height
                boxes.append([class_id, xmin, ymin, xmax, ymax])
        return tf.convert_to_tensor(boxes, dtype=tf.float32)

    labels = tf.py_function(func=parse_labels, inp=[label_path], Tout=tf.float32)

    # Adjust labels to match padded image coordinates
    labels = adjust_bboxes_to_padding(labels, height, width, new_height, new_width)

    # Set shapes explicitly
    img.set_shape([1024, 1024, 3])
    labels.set_shape([None, 5])  # Assuming labels are [class_id, xmin, ymin, xmax, ymax]
    return img, labels


def adjust_bboxes_to_padding(labels, original_height, original_width, resized_height, resized_width):
    """
    Adjust bounding boxes to account for padding applied during aspect-ratio resizing.
    Args:
        labels (tf.Tensor): Bounding boxes in TensorFlow format.
        original_height (float): Original image height.
        original_width (float): Original image width.
        resized_height (float): Resized image height.
        resized_width (float): Resized image width.
    Returns:
        tf.Tensor: Adjusted bounding boxes.
    """
    # Calculate padding applied to make the image 1024x1024
    y_pad = (1024 - resized_height) / 2
    x_pad = (1024 - resized_width) / 2

    # Adjust bounding boxes
    adjusted_bboxes = []
    for label in labels:
        class_id, xmin, ymin, xmax, ymax = label.numpy()
        xmin = xmin * resized_width / original_width + x_pad
        xmax = xmax * resized_width / original_width + x_pad
        ymin = ymin * resized_height / original_height + y_pad
        ymax = ymax * resized_height / original_height + y_pad
        adjusted_bboxes.append([class_id, xmin, ymin, xmax, ymax])

    return tf.convert_to_tensor(adjusted_bboxes, dtype=tf.float32)


def load_dataset(image_dir, label_dir):
    """
    Load and preprocess the dataset for TensorFlow.
    
    Args:
        image_dir (str): Path to the images directory.
        label_dir (str): Path to the labels directory.
    
    Returns:
        tf.data.Dataset: A dataset ready for training or validation.
    """
    # Get the list of image and label files
    image_files = tf.data.Dataset.list_files(f"{image_dir}/*.jpg")
    label_files = tf.data.Dataset.list_files(f"{label_dir}/*.txt")

    # Zip the image and label datasets together
    dataset = tf.data.Dataset.zip((image_files, label_files))

    # Apply the preprocessing function
    dataset = dataset.map(
        lambda img, lbl: tf.py_function(func=preprocess, inp=[img, lbl], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch and prefetch the dataset
    dataset = dataset.batch(16).prefetch(tf.data.AUTOTUNE)  # Adjust batch size as needed
    return dataset


def build_cnn(input_shape=(224, 224, 3), num_classes=5):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # For classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # Load train and validation datasets
    train_dataset = load_dataset("/run/media/krw/PortableSSD/fall_2024_augmented_dataset/tensorflow_dataset/train/images", "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/tensorflow_dataset/train/labels")
    val_dataset = load_dataset("/run/media/krw/PortableSSD/fall_2024_augmented_dataset/tensorflow_dataset/val/images", "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/tensorflow_dataset/val/labels")

    cnn_model = build_cnn(input_shape=(1024, 1024, 3))
    cnn_model.summary()

    # Train the model
    history = cnn_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10
    )


    # Evaluate the model with test data
    test_dataset = load_dataset("/run/media/krw/PortableSSD/fall_2024_augmented_dataset/tensorflow_dataset/test/images", "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/tensorflow_dataset/test/labels")

    # Evaluate the model
    test_loss, test_accuracy = cnn_model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_accuracy:.2f}")

