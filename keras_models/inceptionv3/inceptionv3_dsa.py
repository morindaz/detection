#!/usr/bin/env python
# -*- coding: utf-8 -*-import sys
import sys
from sklearn.metrics import log_loss, confusion_matrix

sys.path.append('../../DenseNet-Keras')

from keras_models.vgg.load_data import load_data
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
from sklearn.metrics import log_loss, accuracy_score

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 1
    num_classes = 13
    batch_size = 16
    nb_epoch = 20
    train_ratio = 0.7

    X_train, Y_train, X_valid, Y_valid,_ = load_data('/home/maoshunyi/moyamoya/test_image', '/home/maoshunyi/moyamoya/train_img/ap_label.xlsx', (img_rows, img_cols), num_classes, train_ratio)
    # Load our model

    image_size = 224
    inceptionV3_conv = InceptionV3(weights=None,include_top=False, input_shape=(image_size, image_size, 1), classes=num_classes)
    print(inceptionV3_conv.summary())

    # Freeze the layers except the last 4 layers
    # for layer in inceptionV3_conv.layers[:-4]:
    #     layer.trainable = False
    #     inceptionV3_conv.layers[0].trainable = True
    # Check the trainable status of the individual layers
    for layer in inceptionV3_conv.layers:
        print(layer, layer.trainable)

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(inceptionV3_conv)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(13, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    model.summary()
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
    score = model.evaluate(X_valid, Y_valid, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy;', score[1])
    print(score)
    import numpy as np

    Y_valid_label = np.argmax(Y_valid, axis=1)
    predictions_valid_label = np.argmax(predictions_valid, axis=1)
    print(confusion_matrix(y_true=Y_valid_label, y_pred=predictions_valid_label))
    with open('result.txt', 'w') as result:
        # result.write(confusion_matrix(y_true=Y_valid_label, y_pred=predictions_valid_label))
        for i in range(len(Y_valid_label)):
            if Y_valid_label[i] != predictions_valid_label[i]:
                result.write('image:{0}\t real:{1}\tpredict:{2}\n'.format(Y_img_name[i], Y_valid_label[i],
                                                                          predictions_valid_label[i]))
