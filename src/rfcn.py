import tensorflow as tf
# import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Dropout, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
import json

# import tensorflow_dataset_loader
from tensorflow_dataset_loader import *

BATCH_SIZE = 8
MAX_BOXES = 100


def custom_bbox_loss(y_true, y_pred):
#     """
#     Compute loss for bounding box predictions, ignoring padded values (-1).
#     """
    mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)  # Mask for valid bounding boxes
    squared_diff = tf.square((y_true - y_pred) * mask)
    return tf.reduce_sum(squared_diff) / tf.reduce_sum(mask)

def build_rfcn(input_shape=(1024, 1024, 3), max_boxes=100):
    # Input layers
    image_input = Input(shape=input_shape)
    roi_input = Input(shape=(max_boxes, 4))  # Bounding box proposals (ymin, xmin, ymax, xmax)
    
    
    # Backbone: Shared feature extractor (e.g., ResNet)
    backbone = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)
    feature_map = backbone.output

    # Position-Sensitive Score Maps (for classification and bounding boxes)
    k = 3  # Grid size for position-sensitive score maps
    class_maps = [Conv2D(1, (1, 1), activation='sigmoid')(feature_map) for _ in range(k * k)]
    bbox_maps = [Conv2D(4, (1, 1), activation='linear')(feature_map) for _ in range(k * k)]

    # Stack position-sensitive score maps
    class_maps = tf.stack(class_maps, axis=0)  # Shape: [k*k, H, W, 1]
    bbox_maps = tf.stack(bbox_maps, axis=0)    # Shape: [k*k, H, W, 4]

    # RoI Pooling
    pooled_features = ROIPoolingLayer(k, class_maps, bbox_maps)([feature_map, roi_input])

    # Classification and Bounding Box Head
    pooled_class_features = pooled_features['class']
    pooled_bbox_features = pooled_features['bbox']

    # Classification Output
    class_output = Dense(1, activation='sigmoid', name='class_output')(pooled_class_features)

    # Bounding Box Output
    bbox_predictions = Dense(4, activation='linear', name='bbox_output')(pooled_bbox_features)

    # Model definition
    model = Model(inputs=[image_input, roi_input], outputs={'class_output': class_output, 'bbox_output': bbox_predictions})
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss={
            'class_output': 'binary_crossentropy',  # Binary classification loss
            'bbox_output': custom_bbox_loss         # Custom loss for bounding box regression
        },
        loss_weights={
            'class_output': 1.0,  # Weight for classification loss
            'bbox_output': 1.0    # Weight for bounding box regression
        },
        metrics={
            'class_output': 'accuracy'  # Metric for classification
        }
    )
    return model


class ROIPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, k, class_maps, bbox_maps, **kwargs):
        super(ROIPoolingLayer, self).__init__(**kwargs)
        self.k = k
        self.class_maps = class_maps
        self.bbox_maps = bbox_maps
    
    def call(self, inputs):
        feature_map, rois = inputs
        pooled_class = tf.reduce_mean(self.class_maps, axis=[1, 2])
        pooled_bbox = tf.reduce_mean(self.bbox_maps, axis=[1, 2])
        return {'class': pooled_class, 'bbox': pooled_bbox}


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

    for img_batch, lbl_batch in train_dataset.take(1):
        print("Image batch shape:", img_batch.shape)
        print("Label dictionary keys:", lbl_batch.keys())
        print("Class output shape:", lbl_batch['class_output'].shape)
        print("BBox output shape:", lbl_batch['bbox_output'].shape)

    for img, lbls in train_dataset.take(1):
        print(f"Labels type: {type(lbls)}")
        print(f"Labels content: {lbls}")

    #  R-FCN Training
    rfcn_model = build_rfcn(input_shape=(1024, 1024, 3), max_boxes=100)
    rfcn_model.summary()

    checkpoint = ModelCheckpoint(
        filepath='rfcn_best.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )

    # Train the model
    history = rfcn_model.fit(
        train_dataset.map(
            lambda img, lbls: (
                {'input_1': img, 'input_2': lbls['bbox_output']},  # Inputs: image and RoIs
                {'class_output': lbls['class_output'],            # Output: class predictions
                'bbox_output': lbls['bbox_output']}              # Output: bounding box predictions
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        ),
        validation_data=val_dataset.map(
            lambda img, lbls: (
                {'input_1': img, 'input_2': lbls['bbox_output']},  # Inputs: image and RoIs
                {'class_output': lbls['class_output'],            # Output: class predictions
                'bbox_output': lbls['bbox_output']}              # Output: bounding box predictions
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        ),
        epochs=100,
        callbacks=[checkpoint],
        verbose=1
    )

    rfcn_model.save('training_rfcn.h5')

    with open('rfcn_training_history.json', 'w') as f:
        json.dump(history.history, f)

    # Evaluate the model with test data
    test_dataset = load_dataset(
        "./dataset_split/images/test",
        "./dataset_split/labels/test",
        batch_size=8
    )

    # Evaluate the model
    results = rfcn_model.evaluate(test_dataset, batch_size=BATCH_SIZE)

    # save evaluation results to file
    with open('rfcn_evaluation_results.json', 'w') as f:
        json.dump(results, f)

    overall_loss = results[0]
    bbox_loss = results[1]
    class_loss = results[2]
    class_accuracy = results[3]

    print(f"Overall Loss: {overall_loss}")
    print(f"Bounding Box Loss: {bbox_loss}")
    print(f"Class Loss: {class_loss}")
    print(f"Class Accuracy: {class_accuracy}")