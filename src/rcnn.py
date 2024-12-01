import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow_dataset_loader import *


def build_backbone(input_shape=(1024, 1024, 3)):
    backbone = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    feature_map = backbone.output  # Use the output feature map
    return Model(inputs=backbone.input, outputs=feature_map)


def build_rpn(feature_map):
    # 3x3 convolution for region proposal
    rpn_conv = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(feature_map)

    # Objectness score (binary classification: object or not)
    objectness = layers.Conv2D(1, (1, 1), activation="sigmoid", name="rpn_objectness")(rpn_conv)

    # Bounding box regression (anchor adjustments)
    bbox_reg = layers.Conv2D(4, (1, 1), activation="linear", name="rpn_bbox_reg")(rpn_conv)

    return objectness, bbox_reg


def roi_pooling(feature_map, rois, pool_size=(7, 7)):
    """Perform ROI Pooling."""
    pooled_rois = tf.image.crop_and_resize(
        feature_map, rois, tf.zeros_like(rois[:, 0], dtype=tf.int32), pool_size
    )
    return pooled_rois


def build_roi_head(pooled_rois, num_classes=2):
    x = layers.TimeDistributed(layers.Flatten())(pooled_rois)
    x = layers.TimeDistributed(layers.Dense(1024, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Dense(1024, activation="relu"))(x)

    # Classification
    class_logits = layers.TimeDistributed(layers.Dense(num_classes, activation="softmax"), name="roi_class")(x)

    # Bounding box regression
    bbox_reg = layers.TimeDistributed(layers.Dense(4, activation="linear"), name="roi_bbox")(x)

    return class_logits, bbox_reg


def build_faster_rcnn(input_shape=(1024, 1024, 3), num_classes=2, max_rois=100):
    # Backbone
    inputs = tf.keras.Input(shape=input_shape)
    feature_map = build_backbone(input_shape)(inputs)

    # RPN
    rpn_objectness, rpn_bbox = build_rpn(feature_map)

    # Placeholder for RoIs (to be generated dynamically in training)
    rois = tf.keras.Input(shape=(max_rois, 4), name="rois")

    # RoI Pooling
    pooled_rois = roi_pooling(feature_map, rois)

    # RoI Head
    class_logits, bbox_reg = build_roi_head(pooled_rois, num_classes)

    # Build the model
    model = Model(inputs=[inputs, rois], outputs={
        "rpn_objectness": rpn_objectness,
        "rpn_bbox": rpn_bbox,
        "roi_class": class_logits,
        "roi_bbox": bbox_reg,
    })

    return model


# def rpn_loss_objectness(y_true, y_pred):
#     return tf.keras.losses.binary_crossentropy(y_true, y_pred)


def rpn_loss_objectness(y_true, y_pred):
    """
    Compute binary crossentropy loss for RPN objectness.

    Args:
        y_true: Ground truth objectness labels (binary, shape: [batch_size, height, width, 1]).
        y_pred: Predicted objectness logits (shape: [batch_size, height, width, 1]).

    Returns:
        Loss value for RPN objectness.
    """
    # Reshape y_true to match y_pred
    y_true = tf.reshape(y_true, tf.shape(y_pred))
    return tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)


def rpn_loss_bbox(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


def roi_loss_class(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)


def roi_loss_bbox(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


if __name__ == '__main__':
    # Load train and validation datasets
    train_dataset = load_dataset(
        "./dataset_split/images/train",
        "./dataset_split/labels/train",
        batch_size=8
    )
    val_dataset = load_dataset(
        "./dataset_split/images/val",
        "./dataset_split/labels/val",
        batch_size=8
    )
 
    faster_rcnn = build_faster_rcnn(input_shape=(1024, 1024, 3), num_classes=2, max_rois=100)

    faster_rcnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss={
        "rpn_objectness": rpn_loss_objectness,
        "rpn_bbox": rpn_loss_bbox,
        "roi_class": roi_loss_class,
        "roi_bbox": roi_loss_bbox,
    })

    for inputs, outputs in train_dataset.take(1):
        y_pred = faster_rcnn(inputs)
        print(f"Model Outputs: {y_pred.keys()}")
        print(f"Dataset Outputs: {outputs.keys()}")
        print(f"RPN Objectness Ground Truth Shape: {outputs['rpn_objectness'].shape}")
        print(f"RPN Objectness Ground Truth Values: {outputs['rpn_objectness']}")

        for key, val in y_pred.items():
            print(f"{key} Shape: {val.shape}")


    # Train the model
    # history = faster_rcnn.fit(
    #     train_dataset,
    #     validation_data=val_dataset,
    #     epochs=1,
    #     steps_per_epoch=len(train_dataset),
    #     validation_steps=len(val_dataset),
    #     verbose=1
    # )

