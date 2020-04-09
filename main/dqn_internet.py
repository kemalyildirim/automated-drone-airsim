#How to Code Deep Q Learning in Tensorflow (Tutorial)
#https://www.youtube.com/watch?v=3Ggq_zoRGP4

import os 
import tensorflow as tf
import numpy as np

#input dims = enviroment
class DQN(object):
    def __init__(self, learning_rate, n_actions, name, fcl_dims=256,
                input_dims=(224,224,4) chpkpt_dir=("/tmp/")):
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.name = name
        self.fcl_dims = fcl_dims
        self.input_dims = input_dims
        self.sess = tf.Session()
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chpkpt_dir, 'dqn.ckpt')
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
    def build_net(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape = [None, *self.input_dims], name='inputs')
            self.actions = tf.placeholder(tf.float32, shape = [None, self.n_actions], name='action_taken')
            self.q_target = tf.placeholder(tf.float32, shape = [None, *self.input_dims])

            conv1 = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=(8,8), strides=4, name='conv1',
                                    kernel_initalizer=tf.variance_scaling_initalizer(scale=2))
            conv1_activated = tf.nn.relu(conv1)

            conv2 = tf.layers.conv2d(inputs=conv1_activated, filters=64, kernel_size=(4,4), strides=2, name='conv2',
                                    kernel_initalizer=tf.variance_scaling_initalizer(scale=2))
            conv2_activated = tf.nn.relu(conv2)

            conv3 = tf.layers.conv2d(inputs=conv2_activated, filters=128, kernel_size=(3,3) strides=1, name='conv3',
                                    kernel_initalizer=tf.variance_scaling_initalizer(scale=2)))
            conv3_activated = tf.nn.relu(conv3)

            
