from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense

class VGGNet:
    @staticmethod
    def build(input_shape, num_classes):
        # Load the VGG16 pre-trained model
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Freeze the weights of the pre-trained layers
        for layer in base_model.layers:
            layer.trainable = False

        # Create a new model
        model = Sequential()
        
        # Add the VGG16 base model
        model.add(base_model)
        
        # Add custom fully connected layers
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
        return model
