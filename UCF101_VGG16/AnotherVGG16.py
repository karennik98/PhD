# import required libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os


def train():
    # Load the UCF101_sp dataset
    data_dir = 'C:\\Users\\karen\\PhD\\UCF101_VGG16\\UCF-101'
    classes = sorted(os.listdir(data_dir))
    num_classes = len(classes)
    video_size = (224, 224)
    frames_per_video = 10

    # Preprocess the data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=video_size,
        shuffle=True,
        seed=123
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'val'),
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=video_size,
        shuffle=True,
        seed=123
    )

    # Define the VGG16 model with 3D convolutional layers for video processing
    inputs = keras.applications.vgg16.Input(shape=(frames_per_video, *video_size, 3))
    base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(*video_size, 3))

    x = keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling3D((1, 2, 2))(x)

    x = keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling3D((2, 2, 2))(x)

    x = keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling3D((2, 2, 2))(x)

    x = keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling3D((2, 2, 2))(x)

    x = keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling3D((2, 2, 2))(x)

    # Flatten and add dense layers for classification
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create a new model with modified layers
    model = keras.models.Model(inputs, outputs)

    # Freeze base VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile and fit the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_ds, validation_data=val_ds, epochs=32)

    model_json = model.to_json()
    with open("C:\\Users\\karen\\PhD\\UCF101_VGG16\\Model\\model_architecture.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("C:\\Users\\karen\\PhD\\UCF101_VGG16\\Model\\model_weights.h5")