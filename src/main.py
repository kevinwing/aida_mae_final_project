import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape
import cv2
import numpy as np

import albumentations as A

from tensorflow_dataset_loader import *
# import tensorflow_dataset_loader


# Albumentations transform for resizing
# transform = A.Compose(
#     [
#         A.Resize(1024, 1024, always_apply=True)
#     ],
#     bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
# )

BATCH_SIZE = 8
MAX_BOXES = 100

# def preprocess_with_albumentations(image_path, label_path):
#     """
#     Preprocess a single image and its corresponding labels using Albumentations.
#
#     Args:
#         image_path (tf.Tensor): Path to the image file.
#         label_path (tf.Tensor): Path to the label file.
#
#     Returns:
#         tf.Tensor, tf.Tensor: Preprocessed image and labels.
#     """
#     # Load the image
#     image_path_str = image_path.numpy().decode("utf-8")
#     image = cv2.imread(image_path_str)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Load labels in TensorFlow format
#     def load_labels(label_file):
#         bboxes = []
#         class_labels = []
#         with open(label_file.numpy().decode("utf-8"), "r") as f:
#             for line in f:
#                 class_id, xmin, ymin, xmax, ymax = map(float, line.strip().split())
#                 bboxes.append([xmin, ymin, xmax, ymax])  # Pascal VOC format
#                 class_labels.append(int(class_id))
#         return bboxes, class_labels
#
#     bboxes, class_labels = load_labels(label_path)
#
#     # Apply Albumentations transform
#     augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
#     resized_image = augmented["image"]
#     resized_bboxes = augmented["bboxes"]
#     resized_labels = augmented["class_labels"]
#
#     # Combine class IDs with resized bounding boxes
#     tf_bboxes = []
#     for i, bbox in enumerate(resized_bboxes):
#         xmin, ymin, xmax, ymax = bbox
#         class_id = resized_labels[i]
#         tf_bboxes.append([class_id, xmin, ymin, xmax, ymax])
#
#     # Normalize image
#     resized_image = resized_image.astype(np.float32) / 255.0
#
#     return resized_image, tf.convert_to_tensor(tf_bboxes, dtype=tf.float32)


# def preprocess(image_path, label_path):
#     """
#     Wrapper to handle Albumentations preprocessing in TensorFlow.
#     """
#     image, labels = tf.py_function(
#         func=preprocess_with_albumentations,
#         inp=[image_path, label_path],
#         Tout=(tf.float32, tf.float32)
#     )
#     image.set_shape([1024, 1024, 3])  # Ensure TensorFlow recognizes the shape
#     labels.set_shape([None, 5])       # [class_id, xmin, ymin, xmax, ymax]
#     return image, labels


# def load_dataset(image_dir, label_dir):
#     """
#     Load and preprocess the dataset using Albumentations.
#
#     Args:
#         image_dir (str): Path to the images directory.
#         label_dir (str): Path to the labels directory.
#
#     Returns:
#         tf.data.Dataset: A dataset ready for training or validation.
#     """
#     # Get the list of image and label files
#     image_files = tf.data.Dataset.list_files(f"{image_dir}/*.jpg")
#     label_files = tf.data.Dataset.list_files(f"{label_dir}/*.txt")
#
#     # Zip the image and label datasets together
#     dataset = tf.data.Dataset.zip((image_files, label_files))
#
#     # Apply the preprocessing function
#     dataset = dataset.map(
#         lambda img, lbl: preprocess(img, lbl),
#         num_parallel_calls=tf.data.AUTOTUNE
#     )
#
#     # Batch and prefetch the dataset
#     dataset = dataset.padded_batch(
#         BATCH_SIZE,
#         padded_shapes=(
#             [1024, 1024, 3],
#             [None, 5]
#         ),
#         padding_values=(
#             0.0,
#             -1.0
#         )
#     ).prefetch(tf.data.AUTOTUNE)  # Adjust batch size as needed
#
#     return dataset


def custom_bbox_loss(y_true, y_pred):
    """
    Compute loss for bounding box predictions, ignoring padded values (-1).
    """
    mask = K.cast(K.not_equal(y_true, -1.0), K.floatx())  # Mask for valid bounding boxes
    squared_diff = K.square((y_true - y_pred) * mask)
    return K.sum(squared_diff) / K.sum(mask)


# def custom_bbox_loss(y_true, y_pred):
#     mask = K.cast(K.not_equal(y_true, -1.0), K.floatx())
#     squared_diff = K.square((y_true - y_pred) * mask)
#     return K.sum(squared_diff) / K.sum(mask)


def build_cnn(input_shape=(1024, 1024, 3), num_classes=1, max_boxes=50):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Classification output: 1 prediction per image
    class_output = Dense(1, activation='sigmoid', name='class_output')(x)

    # Bounding box output: 50 predictions per image (50 boxes, each with 4 values)
    bbox_output = Dense(max_boxes * 4, activation='linear', name='bbox_output')(x)
    bbox_output = Reshape((max_boxes, 4), name='reshape_bbox')(bbox_output)

    model = Model(inputs=inputs, outputs={'class_output': class_output, 'bbox_output': bbox_output})

    model.compile(
        optimizer='adam',
        loss={
            'class_output': 'binary_crossentropy',
            'bbox_output': custom_bbox_loss
        },
        metrics={
            'class_output': 'accuracy',
            # 'bbox_output': None,
        }
    )
    return model


# def build_cnn(input_shape=(1024, 1024, 3), num_classes=1):
#     inputs = Input(shape=input_shape)
#
#     x = Conv2D(32, (3, 3), activation='relu')(inputs)
#     x = MaxPooling2D((2, 2),)(x)
#     x = Conv2D(64, (3, 3), activation='relu')(x)
#     x = MaxPooling2D((2, 2))(x)
#     x = Flatten()(x)
#
#     class_output = Dense(1, activation='sigmoid', name='class_output')(x)
#
#     bbox_output = Dense(4, activation='linear', name='bbox_output')(x)
#
#     model = Model(inputs=inputs, outputs=[class_output, bbox_output])
#
#     model.compile(
#         optimizer='adam',
#         loss={
#             'class_output': 'binary_crossentropy',
#             # 'bbox_output': custom_bbox_loss
#             'bbox_output': 'mse'
#         },
#         metrics={'class_output': 'accuracy'}
#     )
#
#     return model
#

if __name__ == '__main__':
    # Load train and validation datasets
    train_dataset = load_dataset(
        "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/images/train",
        "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/labels/train",
        batch_size=8
    )
    val_dataset = load_dataset(
        "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/images/val",
        "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/labels/val",
        batch_size=8
    )

    for img_batch, lbl_batch in train_dataset.take(1):
        print("Image batch shape:", img_batch.shape)
        print("Label dictionary keys:", lbl_batch.keys())
        print("Class output shape:", lbl_batch['class_output'].shape)
        print("BBox output shape:", lbl_batch['bbox_output'].shape)

    # for image_batch, label_batch in train_dataset.take(1):
    #     print(f"Image batch shape: {image_batch.shape}")  # Expected: (batch_size, 1024, 1024, 3)
    #     print(f"Class labels shape: {label_batch['class_output'].shape}")  # Expected: (batch_size, 50)
    #     print(f"BBox labels shape: {label_batch['bbox_output'].shape}")  # Expected: (batch_size, 50, 4)

    for img, lbls in train_dataset.take(1):
        print(f"Labels type: {type(lbls)}")
        print(f"Labels content: {lbls}")
    # for image, labels in train_dataset.take(1):
    #     print(f"Image batch shape: {image.shape}")
    #     print(f"Label batch shape: {labels.shape}")

    cnn_model = build_cnn(input_shape=(1024, 1024, 3), num_classes=1, max_boxes=MAX_BOXES)
    cnn_model.summary()

    # Train the model
    history = cnn_model.fit(
        train_dataset.map(
            lambda img, lbls: (img, {
                'class_output': lbls['class_output'], 
                'bbox_output': lbls['bbox_output']
            }),
            num_parallel_calls=tf.data.AUTOTUNE
        ),
        validation_data=val_dataset.map(
            lambda img, lbls: (img, {
                'class_output': lbls['class_output'],
                'bbox_output': lbls['bbox_output']
            }),
            num_parallel_calls=tf.data.AUTOTUNE
        ),
        epochs=10
    )

    # history = cnn_model.fit(
    #     train_dataset.map(lambda img, lbls: (img, {'class_output': labls[0], 'bbox_output':})),
    #     validation_data=val_dataset,
    #     epochs=10
    # )


    # Evaluate the model with test data
    test_dataset = load_dataset("/run/media/krw/PortableSSD/fall_2024_augmented_dataset/tensorflow_dataset/test/images", "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/tensorflow_dataset/test/labels")

    # Evaluate the model
    test_loss, test_accuracy = cnn_model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_accuracy:.2f}")

