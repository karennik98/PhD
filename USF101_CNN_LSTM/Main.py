import ModelLoad
import cv2
import numpy as np

def load_video_as_np_array(video_path):
   # Define the shape of the input video frames
    input_shape = (60, 60, 3)

    # Read video file
    cap = cv2.VideoCapture(video_path)
    sequence_length=5
    # Check if video file was opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #Calculate the interval after which frames will be added to the list
    skip_interval = max(int(frame_count/sequence_length), 1)
    # Read video frames as numpy arrays
    frames = []
    for counter in range(sequence_length):
        #Set the current frame postion of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, counter * skip_interval)
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.imwrite("C:\\Users\\karen\\PhD\\USF101_CNN_LSTM\\TMP\\fr" + str(counter) + ".png", frame)
        # cv2.imshow("fr", frame)
        # cv2.waitKey(1)
        frame = cv2.resize(frame, input_shape[:2])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    # Convert frames list to numpy array
    print("len(frames): ", len(frames))
    video_data = np.array(frames)
    # Reshape the input data to match the expected input shape of the model
    video_data = np.reshape(video_data, (1, 5, 60, 60, 3))

    # Release video file
    cap.release()
    return video_data

if __name__ == '__main__':
    video_data = load_video_as_np_array("C:\\Users\\karen\\PhD\\UCF101_VGG16\\UCF-101\\WalkingWithDog\\v_WalkingWithDog_g01_c02.avi")
    model = ModelLoad.load()
    predictions = ModelLoad.predict(model, video_data)

        # Print the predictions for each frame
    for i in range(len(predictions)):
        print('Frame {}: {}'.format(i, predictions[i]))