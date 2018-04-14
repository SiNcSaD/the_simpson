# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# import h5py
# import time
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder

import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def create_model_conv(input_shape, num_classes, show_summary=False):
    """
    CNN Keras model with n convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', name='conv1', input_shape=input_shape))
    model.add(Activation('relu'))
    
    model.add(Conv2D(32, (3, 3), name='conv2'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), padding='same', name='conv3'))
    model.add(Activation('relu'))
    
    model.add(Conv2D(64, (3, 3), name='conv4'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), padding='same', name='conv5'))
    model.add(Activation('relu'))
    
    model.add(Conv2D(128, (3, 3), name='conv6'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))   

    model.add(Conv2D(256, (3, 3), padding='same', name='conv7')) 
    model.add(Activation('relu'))
    
    model.add(Conv2D(256, (3, 3), name='conv8'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    if show_summary:
        print(model.summary())
    
    return model

def training(model, X_train, X_test, y_train, y_test, batch_size, epochs, callbacks_list=[], data_aug=False):
    """
    Training.
    :param model: Keras sequential model
    :param data_augmentation: boolean for data_augmentation (default:True)
    :param callback: boolean for saving model checkpoints and get the best saved model
    :param six_conv: boolean for using the 6 convs model (default:False, so 4 convs)
    :return: model and epochs history (acc, loss, val_acc, val_loss for every epoch)
    """
    if data_aug:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,
            )
        datagen.fit(X_train)
        history = model.fit_generator(datagen.flow(X_train, y_train,
                                                   batch_size=batch_size),
                                      steps_per_epoch=X_train.shape[0] // batch_size,
                                      epochs=epochs,
                                      validation_data=(X_test, y_test),
                                      callbacks=callbacks_list)        
    else:
        history = model.fit(x=X_train, y=y_train,
                            batch_size=batch_size, 
                            epochs=epochs, 
                            shuffle=True,
                            validation_data=(X_test, y_test), 
                            callbacks=callbacks_list)
        
    return model, history