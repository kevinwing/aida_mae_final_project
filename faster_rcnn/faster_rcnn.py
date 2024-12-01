import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D

# Import the YOLO dataset loader
from tensorflow_dataset_loader import load_yolo_dataset
import json


def build_backbone(input_shape):
    """
    Build a simple backbone for feature extraction.

    Args:
        input_shape: Shape of the input image.

    Returns:
        A Keras model for the backbone.
    """
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    return Model(inputs, x, name="backbone")


def build_rpn(feature_map):
    """
    Build the Region Proposal Network (RPN).

    Args:
        feature_map: The output of the backbone.

    Returns:
        rpn_objectness: Binary classification logits for objectness.
        rpn_bbox: Bounding box regression predictions.
    """
    rpn_conv = Conv2D(512, (3, 3), activation='relu', padding='same')(feature_map)

    # Objectness logits
    rpn_objectness = Conv2D(1, (1, 1), activation='sigmoid', name="rpn_objectness")(rpn_conv)

    # Bounding box regression predictions
    rpn_bbox = Conv2D(4, (1, 1), activation='linear', name="rpn_bbox")(rpn_conv)

    return rpn_objectness, rpn_bbox


def build_roi_head(pooled_rois, num_classes):
    """
    Build the RoI head for classification and bounding box regression.

    Args:
        pooled_rois: RoI-pooled features.
        num_classes: Number of object classes.

    Returns:
        class_logits: Class predictions for each RoI.
        bbox_reg: Bounding box regression for each RoI.
    """
    x = GlobalAveragePooling2D()(pooled_rois)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)

    # Class logits
    class_logits = Dense(num_classes, activation='softmax', name="roi_class")(x)

    # Bounding box regression
    bbox_reg = Dense(num_classes * 4, activation='linear', name="roi_bbox")(x)

    return class_logits, bbox_reg


def build_faster_rcnn(input_shape=(1024, 1024, 3), num_classes=2):
    """
    Build the Faster R-CNN model.

    Args:
        input_shape: Shape of the input image.
        num_classes: Number of object classes.

    Returns:
        A Keras Model for Faster R-CNN.
    """
    # Backbone
    inputs = Input(shape=input_shape, name="input_1")
    feature_map = build_backbone(input_shape)(inputs)

    # RPN
    rpn_objectness, rpn_bbox = build_rpn(feature_map)

    # RoIs Input (bounding boxes)
    rois = Input(shape=(None, 4), name="input_2")

    # RoI Pooling (Placeholder for actual RoI pooling logic)
    pooled_rois = tf.image.crop_and_resize(
        feature_map,
        boxes=rois,
        box_indices=tf.zeros((tf.shape(rois)[0],), dtype=tf.int32),
        crop_size=(7, 7)
    )

    # RoI Head
    roi_class_logits, roi_bbox_reg = build_roi_head(pooled_rois, num_classes)

    # Build Faster R-CNN Model
    model = Model(
        inputs=[inputs, rois],
        outputs={
            "rpn_objectness": rpn_objectness,
            "rpn_bbox": rpn_bbox,
            "roi_class": roi_class_logits,
            "roi_bbox": roi_bbox_reg,
        }
    )

    return model


def train_faster_rcnn():
    """
    Train the Faster R-CNN model using the YOLO-formatted dataset.
    """
    # Load dataset
    train_dataset = load_yolo_dataset(
        "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/images/train",
        "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/labels/train",
        batch_size=8
    )
    val_dataset = load_yolo_dataset(
        "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/images/val",
        "/run/media/krw/PortableSSD/fall_2024_augmented_dataset/dataset_split/labels/val",
        batch_size=8
    )

    # Build model
    faster_rcnn = build_faster_rcnn()

    # Compile model
    faster_rcnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "rpn_objectness": tf.keras.losses.BinaryCrossentropy(from_logits=False),
            "rpn_bbox": tf.keras.losses.MeanSquaredError(),
            "roi_class": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            "roi_bbox": tf.keras.losses.MeanSquaredError(),
        },
        metrics={
            "rpn_objectness": "accuracy",
            "roi_class": "accuracy",
        }
    )

    # Train model
    history = faster_rcnn.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        verbose=1
    )

    # Save the model
    faster_rcnn.save("faster_rcnn_model.h5")

    return history


# Run training
if __name__ == "__main__":
    history = train_faster_rcnn()

    with open('faster_rcnn_results.json', 'w') as f:
        json.dump(history.history, f)


