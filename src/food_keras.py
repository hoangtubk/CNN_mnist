"""
 * Created by PyCharm.
 * User: tuhoangbk
 * Date: 11/04/2018
 * Time: 10:29
 * Have a nice day　:*)　:*)
"""

import keras
from keras.applications.vgg16 import VGG16
import cv2 as cv
import numpy as np
import os
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

BATCH_SIZE = 4
EPOCHS = 10
NUMBER_CLASS = 3
IM_SHAPE = (224, 224)
IM_CHANEL = 3

def create_data(path_data, path_np_data, path_np_label, num_class):
    """

    :param path_data: path of raw image data
    :param path_np_data: output path to array data
    :param path_np_label: output path to array label
    :return: None
    """
    iter_class = 0
    for root, dirs, dir_class in os.walk(path_data):
        print(path_data)
        print(dir_class)
        if len(dir_class) == 0:
            continue
        im_data = np.zeros([len(dir_class), IM_SHAPE[0], IM_SHAPE[1], IM_CHANEL])
        im_label = np.zeros([len(dir_class),num_class])
        im_label[:, iter_class] = 1
        for n in range(0, len(dir_class)):
            if not dir_class[n].endswith('.ppm'):
                continue
            img = cv.imread(os.path.join(root, dir_class[n]), 1)
            im_rs = cv.resize(img, IM_SHAPE)
            im_data[n] = im_rs
        np.save(os.path.join(path_np_data, 'data_' + str(iter_class)), im_data)
        np.save(os.path.join(path_np_label, 'label_' + str(iter_class)), im_label)

        iter_class = iter_class + 1

    return

def load_data(path_np_data, path_np_label, num_class, percent):
    """
    Load data from image array
    :param path_np_data: path of image array
    :param path_np_label: path of label array
    :param num_class: number of class
    :param percent: percent = train/test
    :return: Train & Test data after shuffle
    """
    x_train, y_train, x_test, y_test = 0, 0, 0, 0
    for i in range(0, num_class):
        tmp_data = np.load(os.path.join(path_np_data, 'data_' + str(i) + '.npy'))
        tmp_label = np.load(os.path.join(path_np_label, 'label_' + str(i) + '.npy'))
        num_train = (int(percent * tmp_data.shape[0]))
        np.random.shuffle(tmp_data)
        if i == 0:
            x_train = tmp_data[:num_train]
            x_test = tmp_data[num_train:]
            y_train = tmp_label[:num_train]
            y_test = tmp_label[num_train:]
        else:
            x_train1 = tmp_data[:num_train]
            x_test1 = tmp_data[num_train:]
            y_train1 = tmp_label[:num_train]
            y_test1 = tmp_label[num_train:]
            #concatenate
            x_train = np.concatenate((x_train, x_train1))
            x_test = np.concatenate((x_test, x_test1))
            y_train = np.concatenate((y_train, y_train1))
            y_test = np.concatenate((y_test, y_test1))
    # Shuffle training data
    train = np.c_[x_train.reshape(len(x_train), -1), y_train.reshape(len(y_train), -1)]
    np.random.shuffle(train)
    x_train_sf = train[:, :x_train.size//len(x_train)].reshape(x_train.shape)
    y_train_sf = train[:, x_train.size//len(x_train):].reshape(y_train.shape)
    # Shuffle testing data
    test = np.c_[x_test.reshape(len(x_test), -1), y_test.reshape(len(y_test), -1)]
    np.random.shuffle(test)
    x_test_sf = test[:, :x_test.size // len(x_test)].reshape(x_test.shape)
    y_test_sf = test[:, x_test.size // len(x_test):].reshape(y_test.shape)

    return x_train_sf, y_train_sf, x_test_sf, y_test_sf

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(IM_SHAPE[0], IM_SHAPE[1], IM_CHANEL)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUMBER_CLASS))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model

def build_VGG():
    base_model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(IM_SHAPE[0], IM_SHAPE[1], IM_CHANEL))
    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dropout(0.5))
    add_model.add(Dense(256, activation='relu'))
    add_model.add(Dropout(0.5))
    add_model.add(Dense(NUMBER_CLASS))
    add_model.add(Activation('softmax'))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    return model
if __name__ == '__main__':
    # path_data_train_test = '../data_train_test_food'
    # path_data_train_test = '../data_train_test_traffic'
    path_data_train_test = '../data_train_test_AIVerify'
    # path_raw_data = '/media/tuhoangbk/DATA/BigData/food-101/10class'
    # path_raw_data = '/media/tuhoangbk/DATA/BigData/GTSRB/Final_Training/Images'
    path_raw_data = '/media/tuhoangbk/DATA/AI Verify/AnhTestIT'
    # create_data(path_data = path_raw_data,
    #             path_np_data = '../data',
    #             path_np_label= '../label',
    #             num_class=NUMBER_CLASS)
    # x_train_sf, y_train_sf, x_test_sf, y_test_sf = load_data(path_np_data='../data',
    #                                                          path_np_label='../label',
    #                                                          num_class=NUMBER_CLASS,
    #                                                          percent=0.8)
    # np.save(os.path.join(path_data_train_test, 'x_train.npy'), x_train_sf)
    # np.save(os.path.join(path_data_train_test, 'y_train.npy'), y_train_sf)
    # np.save(os.path.join(path_data_train_test, 'x_test.npy'), x_test_sf)
    # np.save(os.path.join(path_data_train_test, 'y_test.npy'), y_test_sf)
    # assert False
    x_train_sf = np.load(os.path.join(path_data_train_test, 'x_train.npy'))
    y_train_sf = np.load(os.path.join(path_data_train_test, 'y_train.npy'))
    x_test_sf = np.load(os.path.join(path_data_train_test, 'x_test.npy'))
    y_test_sf = np.load(os.path.join(path_data_train_test, 'y_test.npy'))

    x_train_sf = x_train_sf.astype('float32')
    x_test_sf = x_test_sf.astype('float32')
    x_train_sf /= 255
    x_test_sf /= 255

    print(x_train_sf.shape)
    print(y_train_sf.shape)
    print(x_test_sf.shape)
    print(y_test_sf.shape)
    model = build_VGG()
    # checkpoint
    filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # fit model
    pre_model = model.fit(x_train_sf, y_train_sf,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          validation_data=(x_test_sf, y_test_sf),
                          callbacks=callbacks_list,
                          shuffle=False)
    plt.plot(pre_model.history['acc'])
    plt.plot(pre_model.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(pre_model.history['loss'])
    plt.plot(pre_model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

