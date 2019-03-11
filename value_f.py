import tensorflow as tf 
import numpy as np

from nn_utils import get_FC_model

class ValueFunction():

    def __init__(self, input_dim=10):
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, input_dim))
        self.value = tf.placeholder(dtype=tf.float32, shape=(None, 1)) #value tensor (sorta like label in supervised)
        self.network = get_FC_model(self.x, "value_network", True, hidden_size=[128, 128], output_size=1)
        
        self.loss_op = tf.losses.huber_loss(self.value, self.network)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001) #TODO decide what optimizer to use
        self.train_op = self.optimizer.minimize(self.loss_op)
    
    #Takes states and values, trains, returns loss
    def train(self, sess, states, values):
        _, output = sess.run((self.train_op, self.loss_op), feed_dict={self.x:states, self.value:values})
        return output 

    #Takes states and does one forward pass of the network, no training
    def forward(self, sess, states):
        values = sess.run(self.network, feed_dict={self.x:states})
        return values



            

























