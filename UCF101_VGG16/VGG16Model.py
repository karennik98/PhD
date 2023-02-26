# import required libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd

class VGG16Model:
    def prepareForTrain(self):
        # set up data directories
        self.train_dir = '/UCF101_sp\\train'
        # self.val_dir = ''
        self.test_dir = '/UCF101_sp\\test'

        # set up data generators
        self.batch_size = 32
        self.train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        self.train_generator = self.train_datagen.flow_from_directory(self.train_dir, target_size=(224, 224),
                                                                      batch_size=self.batch_size)
        print(len(self.train_generator))

        # self.val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        # self.val_generator = self.val_datagen.flow_from_directory(self.val_dir, target_size=(224, 224), batch_size=self.batch_size)

        self.test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        self.test_generator = self.test_datagen.flow_from_directory(self.test_dir, target_size=(224, 224),
                                                                    batch_size=self.batch_size)

    def train(self):
        # set up the VGG16 Model
        vgg = keras.applications.vgg16.VGG16(input_shape=(224, 224, 3), include_top=False)
        for layer in vgg.layers:
            layer.trainable = False
        x = keras.layers.Flatten()(vgg.output)
        x = keras.layers.Dense(101, activation='softmax')(x)
        model = keras.models.Model(vgg.input, x)

        # compile the Model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # train the Model
        epochs = 32
        steps_per_epoch = len(self.train_generator)
        print(len(self.train_generator))
        # val_steps = len(self.val_generator)
        model.fit(self.train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

        # evaluate the Model on the test set
        test_steps = len(self.test_generator)
        test_loss, test_acc = model.evaluate(self.test_generator, steps=test_steps)
        print(f'Test accuracy: {test_acc}')

        model_json = model.to_json()
        with open("C:\\Users\\karen\\PhD\\UCF101_VGG16\\Model\\model_architecture.json", "w") as json_file:
            json_file.write(model_json)

        model.save_weights("C:\\Users\\karen\\PhD\\UCF101_VGG16\\Model\\model_weights.h5")

    def load(self):
        # Load the model architecture from the JSON file
        with open("C:\\Users\\karen\\PhD\\UCF101_VGG16\\Model\\model_architecture.json", "r") as json_file:
            loaded_model_json = json_file.read()
        loaded_model = keras.models.model_from_json(loaded_model_json)

        # Load the model weights from the HDF5 file
        loaded_model.load_weights("C:\\Users\\karen\\PhD\\UCF101_VGG16\\Model\\model_weights.h5")