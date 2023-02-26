import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
import Utils


# Enable GPU memory growth to prevent TensorFlow from allocating all memory on the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

label_data = pd.read_csv(Utils.anotations_path + "classInd.txt", sep=' ', header=None)
label_data.columns=['index', 'labels']
label_data = label_data.drop(['index'], axis=1)
label_data.head()

#Total Number of video folders for classification
len(label_data)

path=[]
for label in label_data.labels.values:
    path.append(Utils.data_path+label+"\\")
print(path[0])

#Function for Feature Extraction
def feature_extraction(video_path):
    width=60
    height=60
    sequence_length=5
    frames_list=[]
    #Read the Video
    video_reader = cv2.VideoCapture(video_path)
    #get the frame count
    frame_count=int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    #Calculate the interval after which frames will be added to the list
    skip_interval = max(int(frame_count/sequence_length), 1)
    #iterate through video frames
    for counter in range(sequence_length):
        #Set the current frame postion of the video
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, counter * skip_interval)
        #Read the current frame
        ret, frame = video_reader.read()
        if not ret:
            break;
        #Resize the image
        frame=cv2.resize(frame, (height, width))
        frame = frame/255
        #Append to the frame
        frames_list.append(frame)
    video_reader.release()
    #Return the Frames List
    return frames_list

#Function for loading video files, Process and store in a data set
def load_video(datasets):
    global image
    label_index=0
    labels=[]
    images=[]
    #Iterate through each foler corresponding to category
    for folder in datasets:
        for file in tqdm(os.listdir(folder)):
            #Get the path name for each video
            video_path = os.path.join(folder, file)
            #Extract the frames of the current video
            frames_list = feature_extraction(video_path)
            images.append(frames_list)
            labels.append(label_index)
        label_index+=1
    return np.array(images, dtype='float16'), np.array(labels, dtype='int8')

#Due to memory allocation problem. I will select last 60 video folders for classification
images, labels = load_video(path[41:])

#Shapes
print(images.shape, pd.Series(labels).shape)

#Train Test Split
x_train, x_test, y_train, y_test=train_test_split(images, labels, test_size=0.06, random_state=10)
print(x_train.shape, x_test.shape, np.array(y_train).shape, np.array(y_test).shape)

model = keras.models.Sequential()

model.add(keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='LeakyReLU', data_format='channels_last',
                     return_sequences=True, recurrent_dropout=0.2,
                     input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], 3)))
model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.2)))

model.add(keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='LeakyReLU', data_format='channels_last',
                     return_sequences=True, recurrent_dropout=0.2))
model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.2)))

model.add(keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='LeakyReLU', data_format='channels_last',
                     return_sequences=True, recurrent_dropout=0.2))
model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.2)))

model.add(keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='LeakyReLU', data_format='channels_last',
                     return_sequences=True, recurrent_dropout=0.2))
model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(60, activation='softmax'))
model.summary()

#compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

#Model training
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

# Train the model on GPU
with tf.device('/GPU:0'):
    history = model.fit(x_train, keras.utils.to_categorical(y_train), batch_size=32, epochs=50, validation_data=(x_test, keras.utils.to_categorical(y_test)), callbacks=[es])

#Plot the graph to check training and testing accuracy over the period of time
plt.figure(figsize=(13,5))
plt.title("Accuracy vs Epochs")
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.legend(loc='best')
plt.show()

y_pred = model.predict(x_test)
predicted_classes=[]
for i in range(len(y_test)):
    predicted_classes.append(np.argmax(y_pred[i]))

#Test Accuracy
accuracy_score(y_test, predicted_classes)

#Confusion Matrix
plt.figure(figsize=(25,25))
plt.title("Confusion matrix")
cm=confusion_matrix(y_test, predicted_classes)
plt.imshow(cm)
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center")
plt.show()

model.save(Utils.model_path)
   