import VGG16Model
import AnotherVGG16
import torch
import tensorflow

if __name__ == '__main__':
    # import tensorflow as tf
    #
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # print(tf.config.list_physical_devices('GPU'))

    # print(torch.cuda.is_available())
    # model = VGG16Model.VGG16Model()
    # model.prepareForTrain()
    # model.train()
    # AnotherVGG16.train()
    from tensorflow.python.client import device_lib


    def get_available_devices():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos]


    print(get_available_devices())
    # my output was => ['/device:CPU:0']
    # good output must be => ['/device:CPU:0', '/device:GPU:0']

