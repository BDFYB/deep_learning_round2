import tensorflow as tf 
from tools.lazy_property_decorator import lazy_property

class LinerRegresion(object):
    def __init__(self):
        self.input = tf.placeholder("float32", shape=(None, 2), name="input")
        self.label = tf.placeholder("float32", shape=(None, 1), name="label")
        self.w = tf.Variable(tf.random_normal([2, 1], stddev=0.35), name="weights")
        self.b = tf.Variable(0., name="bias")
        self.learning_rate = 0.0000055

    @lazy_property
    def prediction(self):
        return tf.matmul(self.input, self.w) + self.b

    @lazy_property
    def loss(self):
        loss = tf.reduce_sum(tf.squared_difference(self.label, self.prediction), name='loss')
        return loss

    @lazy_property
    def optimize(self):
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)


