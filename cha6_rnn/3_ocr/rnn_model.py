#!/bin/python

from tools.data_pool import DataPool
from tools.lazy_property import lazy_property
import tensorflow as tf 

class RnnModule(object):
    def __init__(self, params):
        self.inputs = tf.placeholder(shape=(params.max_word_length, None, params.single_image_length), 
                                     name="inputs",
                                     dtype=tf.float32)
        self.labels = tf.placeholder(shape=(params.max_word_length, None, params.output_classes),
                                     name="labels",
                                     dtype=tf.float32)
        self.params = params
        self.rnn_cell = tf.nn.rnn_cell.GRUCell(params.rnn_hidden_size)

        self.train
        self.optimize
        self.prediction

    @lazy_property
    def train(self):
        # input shape: max_word_length * batch_size * single_image_length
        # output shape: max_word_length * batch_size * rnn_hidden_size
        # 因为一共26类，后面加个softmax层处理
        
        output, state = tf.nn.dynamic_rnn(
            cell=self.rnn_cell, 
            sequence_length=self._seq_length,
            inputs=self.inputs,
            time_major=True, 
            dtype=tf.float32)

        
        # 注意这层之后的处理，为了使输出共享一个softmax分类器，采用先扁平化后反扁平化的方式
        # 可以理解为对一个input_size 为rnn_hidden_size的输入进行 26分类，而
        # 输入batch_size是max_word_length * batch_size 的一个大batch
        # 另外这里要考虑是否gather一下，把为0的output干掉（即填充的部分）
        flattern_output = tf.reshape(
            tensor=output,
            shape=(-1, self.params.rnn_hidden_size)
        )
        #flattern_output shape: (max_word_length * batch_size)  * rnn_hidden_size
        
        """
        weights = tf.Variable(tf.ones([self.params.rnn_hidden_size, self.params.output_classes]), name="fully_connect_weights")
        fully_layer = tf.matmul(flattern_output, weights, name="fully_connect_output")
        """

        # 不用全连接层原因：因为输入有许多填充的0，因此不希望加入basis
        # 可以用原因：在计算loss中使用mask直接搞定非0的填充元素
        
        fully_layer = tf.contrib.layers.fully_connected(
            inputs=flattern_output,
            num_outputs=self.params.output_classes,
            #biases不能为0，过小也不行，否则计算loss的时候求log会出现INF
            biases_initializer=tf.constant_initializer(0.5),
        )
        #flattern_output shape: (max_word_length * batch_size)  * output_classes
        
        # 不进行softmax，出现了发散，之后通过调biases初始值、gradient_clipping值能减缓发散，但最终50步左右都发散了
        softmax_layer = tf.nn.softmax(fully_layer)

        reverse_flattern_output = tf.reshape(
            tensor=softmax_layer,
            shape=(self.params.max_word_length, -1, self.params.output_classes),
        )
        #reverse_flattern_output shape: max_word_length * batch_size  * output_classes
        
        return reverse_flattern_output
        

    @lazy_property
    def loss(self):
        cross_entropy = -tf.reduce_sum(self.labels * tf.log(self.train), axis=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.labels), axis=2))
        cross_entropy *= mask
        loss = tf.reduce_sum(cross_entropy, axis=0)
        loss_final = tf.reduce_mean(loss / tf.cast(self._seq_length, tf.float32))
        return loss_final

    @lazy_property
    def optimize(self):
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=self.params.learning_rate,
            momentum=self.params.momentum,
        )
        gradient = optimizer.compute_gradients(self.loss) 
        if self.params.gradient_clipping:
            limit = self.params.gradient_clipping
            gradient = [
                (tf.clip_by_value(g, -limit, limit), v)
                if g is not None else (None, v)
                for g, v in gradient
            ]
        optimized = optimizer.apply_gradients(gradient)
        return optimized

    @lazy_property
    def _seq_length(self):
        signed_data = tf.sign(tf.abs(self.inputs))
        length = tf.reduce_sum(tf.reduce_max(signed_data, axis=2), axis=0)
        return length

    @lazy_property
    def prediction(self):
        return tf.argmax(self.train, axis=2, name="prediction")

