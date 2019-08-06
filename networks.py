import tensorflow as tf 
import numpy as np 
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D
from tensorflow.nn import relu

#Intent of this file is to provide function to map and x_ph to a y_ph or a v_ph
#some functions do one mapping, some do both by calling single mapping functions

def mlp(x, layers, activation=relu, output_activation=None, name=None):
    for h in layers[:-1]:
        x = Dense(h, activation=activation)(x)
    return Dense(layers[-1], activation=output_activation, name=name)(x)

def cartpole_mlp(x):
    y_ph = mlp(x, layers=(64, 64, 2), name="y_output")
    v_ph = mlp(x, layers=(64, 64, 1), name="value_output")
    return y_ph, v_ph

def general_mlp(x, output_dim=4):
    y_ph = mlp(x, layers=(64, 64, output_dim), name="y_output")
    v_ph = mlp(x, layers=(64, 64, 1), name="value_output")
    return y_ph, v_ph

def quadrotor_mlp_1(x, output_dim=4):
    y_ph = mlp(x, layers=(512, 256, 128, output_dim), name="y_output")
    v_ph = mlp(x, layers=(512, 256, 128, 1), name="value_output")
    return y_ph, v_ph

#TODO decide if this is functional or not 
def double_conv_pool(input, filters=4, kernel_size=4, pool_size=(2,2), activation=relu):
    c1 = Conv2D(filters, kernel_size=kernel_size, padding="same", activation=activation)(input)
    c2 = Conv2D(filters, kernel_size=kernel_size, padding="same", activation=activation)(c1)
    mp1 = MaxPool2D(pool_size=pool_size, padding="same")(c2)
    return BatchNormalization()(mp1)

def convnet_atari_1(x, output_dim, name=None):
    s1 = double_conv_pool(x, filters=64)
    s2 = double_conv_pool(s1, filters=32)
    s3 = double_conv_pool(s2, filters=16)
    fc_input = tf.reshape(s3, [-1, 27*20*16]) #This shaping can probably be done automatically TODO
    fc1 = Dense(32, activation=relu)(fc_input)
    fc2 = Dense(16, activation=relu)(fc1)
    return Dense(output_dim, name=name)(fc2)

def breakout_convnet(x):
    y_ph = convnet_atari_1(x, 6, name="y_output")
    v_ph = convnet_atari_1(x, 1, name="value_output")
    return y_ph, v_ph