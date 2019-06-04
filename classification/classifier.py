import os, sys, pdb

import tensorflow as tf
# import keras
# from keras import layers 
# from keras.models import Model
# from inception_v3 import InceptionV3

from tensorflow import keras 
layers = keras.layers
Model = keras.models.Model

def preprocess_input(x):
    with tf.name_scope('preprocess_input'):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

def get_main_network(name, input_tensor, num_classes, use_weights=False):

    processed_tensor = preprocess_input(input_tensor)

    if name == 'inception':
        base_model = keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet' if use_weights else None,
                                                    pooling='avg',
                                                    input_tensor=processed_tensor)
    # adding the last fully connected layer
    last_output = base_model.output
    cls = layers.Dense(num_classes)
    output = cls(last_output)
    model = Model(inputs=base_model.input, outputs=output)
    model_vars = []
    if use_weights:
        for layer in base_model.layers:
            model_vars += [k for k in layer.weights]
    # print (len(model_vars), ' variables will not be initialized')
    for layer in list(model.layers):
        layer.trainable = True

    # if use_weights:
    #     with tf.Session() as sess:
    #         self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='inception')
    #         tf.train.Saver(self.vgg_weights).save(sess, os.path.join(path, 'inception_init'))
    #         print ('save inception_init for session loading')
    return model, model_vars