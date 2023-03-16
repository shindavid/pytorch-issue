"""
Attempt at porting from torch to keras.

python full_demo.py
"""
import random
import sys

import numpy as np
import tensorflow as tf


num_residual_blocks = 19
if len(sys.argv) > 1:
    num_residual_blocks = int(sys.argv[1])


print(f'Full demo with {num_residual_blocks} residual blocks')


random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


def conv_block(x, n_conv_filters):
    x = tf.keras.layers.Conv2D(n_conv_filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def res_block(x, n_conv_filters):
    identity = x
    x = tf.keras.layers.Conv2D(n_conv_filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(n_conv_filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x += identity  # skip connection
    return tf.nn.relu(x)


def policy_head(x):
    x = tf.keras.layers.Conv2D(2, kernel_size=1, strides=1, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(7)(x)
    return x


def model(shape=(2, 7, 6)):
    x_input = tf.keras.Input(shape=shape)
    x = conv_block(x_input, 64)
    for _ in range(num_residual_blocks):
        x = res_block(x, 64)
    x = policy_head(x)
    return tf.keras.Model(inputs=x_input, outputs=x)


net = model()
net.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=tf.keras.losses.BinaryCrossentropy(),
)
net.fit(
    tf.random.normal((128, 2, 7, 6)),
    tf.random.normal((128, 7)),
    batch_size=32
)


def get_output(batch_size):
    input_tensor = tf.zeros((batch_size, 2, 7, 6))
    output_tensor = net(input_tensor)
    return output_tensor[:1]


out1 = get_output(1)
for b in range(2, 64):
    out = get_output(b)
    print('Batch size {} diffs: {}'.format(b, out - out1))

