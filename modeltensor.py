import tensorflow as tf
import params
import numpy as np


def cre_weight(shape):
    return tf.get_variable("weight", shape, initializer=
    tf.truncated_normal_initializer(dtype= tf.float32, stddev= 1e-1), dtype= tf.float32)

def cre_biases(shape):
    return tf.get_variable("bias", shape, initializer= tf.constant_initializer(0.0), dtype= tf.float32)

def cre_conv(x, w, pad):
    return tf.nn.conv2d(x, w, strides= [1, 1, 1, 1], padding= pad)

def cre_pooling2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding= "VALID")


class LeNet_5(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('conv1_layer'):
            w_conv1 = cre_weight([5, 5, 1, 6])
            b_conv1 = cre_biases([6])
            conv1 = tf.nn.relu(cre_conv(x, w_conv1, "SAME")) #28x28x6
            pool1 = cre_pooling2x2(conv1) #14x14x6

        with tf.variable_scope('conv2_layer'):
            w_conv2 = cre_weight([5, 5, 6, 16])
            b_conv2 = cre_biases([16])
            conv2 = tf.nn.relu(cre_conv(pool1, w_conv2, "VALID")) #10x10x16
            pool2 = cre_pooling2x2(conv2) #5x5x16

        with tf.variable_scope('fc1_layer'):
            shape = int(np.prod(pool2.get_shape()[1:]))
            w_fc1 = cre_weight([shape, 120])
            b_fc1 = cre_biases([120])
            pool2_flat = tf.reshape(pool2, ([-1, shape]))
            fc1 = tf.nn.relu(tf.matmul(pool2_flat, w_fc1) + b_fc1)

        dropout_fc1 = tf.nn.dropout(fc1, keep_prob= self.keep_prob)

        with tf.variable_scope('fc2_layer'):
            w_fc2 = cre_weight([120, 84])
            b_fc2 = cre_biases([84])
            fc2 = tf.nn.relu(tf.matmul(dropout_fc1, w_fc2) + b_fc2)

        dropout_fc2 = tf.nn.dropout(fc2, keep_prob= self.keep_prob)

        with tf.variable_scope('out_layer'):
            w_out = cre_weight([84, 10])
            b_out = cre_biases([10])
            y_out = tf.nn.relu(tf.matmul(dropout_fc2, w_out) + b_out)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels= self.y, logits= y_out))
        self.pred = tf.argmax(y_out, 1)
        self.corr_pred = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_out, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.corr_pred, tf.float32))