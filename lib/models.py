import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet50


def build_standard_cnn(input_shape=(1024, 1024, 3), num_classes=5):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # For classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_fast_rcnn(input_shape=(640, 640, 3), num_classes=5):
    # Base feature extractor
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False  # Freeze pre-trained layers
    
    # R-CNN Head
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    class_output = Dense(num_classes, activation='softmax', name='class_output')(x)
    bbox_output = Dense(4, activation='linear', name='bbox_output')(x)  # For bounding box regression

    model = Model(inputs=base_model.input, outputs=[class_output, bbox_output])
    model.compile(
        optimizer='adam',
        loss={'class_output': 'sparse_categorical_crossentropy', 'bbox_output': 'mse'},
        metrics={'class_output': 'accuracy'}
    )
    return model


def build_rfcn(input_shape=(1024, 1024, 3), num_classes=5):
    # Base feature extractor
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False  # Freeze pre-trained layers
    
    # Fully convolutional output layers
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(base_model.output)
    x = Conv2D(num_classes, (1, 1), activation='softmax', name='class_output')(x)  # Class prediction
    bbox_output = Conv2D(4, (1, 1), activation='linear', name='bbox_output')(x)    # Bounding box regression
    
    # Global pooling for outputs
    class_output = GlobalAveragePooling2D()(x)
    bbox_output = GlobalAveragePooling2D()(bbox_output)

    model = Model(inputs=base_model.input, outputs=[class_output, bbox_output])
    model.compile(
        optimizer='adam',
        loss={'class_output': 'sparse_categorical_crossentropy', 'bbox_output': 'mse'},
        metrics={'class_output': 'accuracy'}
    )
    return model


if __name__ == '__main__':
    # CNN
    # print('CNN Architecture')
    # standard_cnn = build_standard_cnn()
    # standard_cnn.summary()
    # Fast R-CNN
    print('Fast R-CNN Architecture')
    fast_rcnn = build_fast_rcnn()
    fast_rcnn.summary()
    # R-FCN
    # print('R-FCN Architecture')
    # rfcn = build_rfcn()
    # rfcn.summary()

