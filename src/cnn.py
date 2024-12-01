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


def build_cnn(input_shape=(1024, 1024, 3), num_classes=1, max_boxes=100):
    # Load VGG16 as backbone, exluding the top fully connected layers
    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    vgg16_base.trainable = False

    inputs = vgg16_base.input

    # Add layers for classification and bounding box regression
    x = vgg16_base.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Class output: 1 prediction per image
    class_output = Dense(1, activation='sigmoid', name='class_output')(x)

    # Bounding box output: 100 per image
    bbox_output = Dense(max_boxes * 4, activation='linear', name='bbox_output')(x)
    bbox_output = Reshape((max_boxes, 4), name='reshape_box')(bbox_output)

    model = Model(inputs=inputs, outputs={'class_output': class_output, 'bbox_output': bbox_output})

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
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


def build_rfcn(input_shape=(1024, 1024, 3), max_boxes=100):
    # Input layer
    image_input = Input(shape=input_shape)
    
    # Backbone: Shared feature extractor (e.g., ResNet)
    backbone = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)
    feature_map = backbone.output

    # Position-Sensitive Score Maps (for classification and bounding boxes)
    k = 3  # Grid size for position-sensitive score maps
    class_maps = []
    bbox_maps = []
    for _ in range(k * k):
        class_maps.append(Conv2D(1, (1, 1), activation='sigmoid')(feature_map))  # Single class
        bbox_maps.append(Conv2D(4, (1, 1), activation='linear')(feature_map))   # Four coordinates

    # Stack position-sensitive score maps
    class_maps = tf.stack(class_maps, axis=0)  # Shape: [k*k, H, W, 1]
    bbox_maps = tf.stack(bbox_maps, axis=0)    # Shape: [k*k, H, W, 4]

    # RoI Align
    roi_input = Input(shape=(max_boxes, 4))  # Bounding box proposals (ymin, xmin, ymax, xmax)
    pooled_features = Lambda(lambda x: roi_pooling(x[0], x[1], k, class_maps, bbox_maps))([feature_map, roi_input])

    # Classification and Bounding Box Heads
    pooled_class_features = pooled_features['class']
    pooled_bbox_features = pooled_features['bbox']

    # Classification Output
    class_logits = Dense(1, activation='sigmoid', name='class_output')(pooled_class_features)

    # Bounding Box Output
    bbox_predictions = Dense(4, activation='linear', name='bbox_output')(pooled_bbox_features)

    # Model definition
    model = Model(inputs=[image_input, roi_input], outputs={'class_output': class_logits, 'bbox_output': bbox_predictions})
    
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


def roi_pooling(feature_map, rois, k, class_maps, bbox_maps):
    # Perform position-sensitive RoI pooling for classification and bounding boxes
    # Implementation here depends on the grid split (k x k) and RoIs
    # For simplicity, this will return mock pooled features
    return {
        'class': tf.reduce_mean(class_maps, axis=[1, 2]),  # Mock pooling
        'bbox': tf.reduce_mean(bbox_maps, axis=[1, 2]),   # Mock pooling
    }


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


    # CNN model Training
    cnn_model = build_cnn(input_shape=(1024, 1024, 3), num_classes=1, max_boxes=MAX_BOXES)
    cnn_model.summary()

    checkpoint = ModelCheckpoint(
        filepath='cnn_best.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )

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
        epochs=100,
        callbacks=[checkpoint]
    )

    cnn_model.save('training_cnn.h5')

    with open('cnn_training_history.json', 'w') as f:
        json.dump(history.history, f)

    # Evaluate the model with test data
    test_dataset = load_dataset(
        "./dataset_split/images/test",
        "./dataset_split/labels/test",
        batch_size=8
    )

    # Evaluate the model
    results = cnn_model.evaluate(test_dataset, batch_size=BATCH_SIZE)

    # save evaluation results to file
    with open('cnn_evaluation_results.json', 'w') as f:
        json.dump(results, f)

    overall_loss = results[0]
    bbox_loss = results[1]
    class_loss = results[2]
    class_accuracy = results[3]

    print(f"Overall Loss: {overall_loss}")
    print(f"Bounding Box Loss: {bbox_loss}")
    print(f"Class Loss: {class_loss}")
    print(f"Class Accuracy: {class_accuracy}")


    #  R-FCN Training
    model = build_rfcn(input_shape=(1024, 1024, 3), max_boxes=100)
    model.summary()
