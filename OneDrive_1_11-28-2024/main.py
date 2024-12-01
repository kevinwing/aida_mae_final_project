import tensorflow as tf
# import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
import json

# import tensorflow_dataset_loader
from tensorflow_dataset_loader import *

BATCH_SIZE = 8
MAX_BOXES = 100


# def custom_bbox_loss(y_true, y_pred):
#     """
#     Compute loss for bounding box predictions, ignoring padded values (-1).
#     """
#     mask = K.cast(K.not_equal(y_true, -1.0), K.floatx())  # Mask for valid bounding boxes
#     squared_diff = K.square((y_true - y_pred) * mask)
#     return K.sum(squared_diff) / K.sum(mask)


def custom_bbox_loss(y_true, y_pred):
#     """
#     Compute loss for bounding box predictions, ignoring padded values (-1).
#     """
    mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)  # Mask for valid bounding boxes
    squared_diff = tf.square((y_true - y_pred) * mask)
    return tf.reduce_sum(squared_diff) / tf.reduce_sum(mask)


# def build_cnn(input_shape=(1024, 1024, 3), num_classes=1, max_boxes=50):
#     inputs = Input(shape=input_shape)

#     x = Conv2D(32, (3, 3), activation='relu')(inputs)
#     x = MaxPooling2D((2, 2))(x)
#     x = Conv2D(64, (3, 3), activation='relu')(x)
#     x = MaxPooling2D((2, 2))(x)
#     x = Flatten()(x)

#     # Classification output: 1 prediction per image
#     class_output = Dense(1, activation='sigmoid', name='class_output')(x)

#     # Bounding box output: 50 predictions per image (50 boxes, each with 4 values)
#     bbox_output = Dense(max_boxes * 4, activation='linear', name='bbox_output')(x)
#     bbox_output = Reshape((max_boxes, 4), name='reshape_bbox')(bbox_output)

#     model = Model(inputs=inputs, outputs={'class_output': class_output, 'bbox_output': bbox_output})

#     model.compile(
#         optimizer='adam',
#         loss={
#             'class_output': 'binary_crossentropy',
#             'bbox_output': custom_bbox_loss
#         },
#         metrics={
#             'class_output': 'accuracy',
#             # 'bbox_output': None,
#         }
#     )
#     return model


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

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # accuracy = history.history['class_output_accuracy']
    # val_accuracy = history.history['val_class_output_accuracy']

    # plt.figure(figsize=(10, 5))
    # plt.plot(loss, label='Training Loss')    
    # plt.plot(val_loss, label='Validation Loss')
    # plt.title('Loss over epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig('cnn_loss_plot.png')
    # plt.close()
    
    # plt.figure(figsize=(10, 5))
    # plt.plot(loss, label='Training Accuracy')    
    # plt.plot(val_loss, label='Validation Accuracy')
    # plt.title('Accuracy over epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.savefig('cnn_accuracy_plot.png')
    # plt.close()

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
