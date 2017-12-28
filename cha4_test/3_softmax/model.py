import tensorflow as tf 
from tools.lazy_property import lazy_property

class SoftMaxModel(object):
    def __init__(self):
        self.inputs = tf.placeholder('float32', (None, 4), name="inputs")
        self.labels = tf.placeholder('float32', (None, 3), name="labels")
        self.learning_rate = 0.01
        self.prediction
        self.infer

    @lazy_property
    def prediction(self):
        fully_connect_layer = tf.contrib.layers.fully_connected(
            inputs=self.inputs,
            num_outputs=3,
            activation_fn = None,
            weights_initializer = tf.contrib.layers.xavier_initializer(),
            biases_initializer = tf.contrib.layers.xavier_initializer(),            
        )
        return fully_connect_layer

    @lazy_property
    def loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.prediction), name="loss")
        return loss

    @lazy_property
    def optimizer(self):
        optimze = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        return optimize

    @lazy_property
    def infer(self):
        result = tf.nn.softmax(self.prediction, name="softmax")
        inference = tf.argmax(result, 1, name="infer")
        return inference