#!/bin/python

from tools.data_pool import DataPool
from tools.attr_dictionary import AttrDict
from tools.lazy_property import lazy_property
import tensorflow as tf 
import os
import json
import numpy as np

class BidirectionRnnModule(object):
    def __init__(self, params):
        self.inputs = tf.placeholder(shape=(params.max_word_length, None, params.single_image_length), 
                                     name="inputs",
                                     dtype=tf.float32)
        self.labels = tf.placeholder(shape=(params.max_word_length, None, params.output_classes),
                                     name="labels",
                                     dtype=tf.float32)
        self.params = params
        self.rnn_cell = tf.nn.rnn_cell.GRUCell(params.rnn_hidden_size)
        self.global_step = tf.Variable(1, trainable=False, name="global_steps")
        self.learning_rate=tf.train.exponential_decay(
            learning_rate=0.1,
            global_step=self.global_step,
            decay_steps=200,
            decay_rate=0.9,
            staircase=True,
            name="learning_rate",
        )

        self.train
        self.optimize
        self.prediction


    @lazy_property
    def train(self):
        # input shape: max_word_length * batch_size * single_image_length
        # output shape: max_word_length * batch_size * rnn_hidden_size
        forward_output, forward_state = tf.nn.dynamic_rnn(
            cell=self.rnn_cell,
            inputs=self.inputs,
            sequence_length=self._seq_length,
            time_major=True, 
            dtype=tf.float32,
            scope='forward_rnn')

        backward_output, backward_state = tf.nn.dynamic_rnn(
            cell=self.rnn_cell,
            inputs=tf.reverse_sequence(self.inputs, 
                                       seq_lengths=self._seq_length, 
                                       batch_axis=1, seq_axis=0),
            sequence_length=self._seq_length,
            time_major=True, 
            dtype=tf.float32,
            scope='backward_rnn')

        backward_reversed = tf.reverse_sequence(
                    backward_output, 
                    seq_lengths=self._seq_length, 
                    batch_axis=1, seq_axis=0),

        total_output = tf.concat([forward_output, backward_output], axis=2)

        rnn_layer_shape = int(total_output.get_shape()[2])
        reversed_output = tf.reshape(total_output, shape=(-1, rnn_layer_shape))

        #做softmax
        fully_connect_layer = tf.contrib.layers.fully_connected(
            inputs=reversed_output,
            num_outputs=self.params.output_classes,
            biases_initializer=tf.constant_initializer(0.8),
        )
        softmax_layer = tf.nn.softmax(fully_connect_layer)

        reversed_back_output = tf.reshape(softmax_layer, shape=(self.params.max_word_length, -1, self.params.output_classes))

        return reversed_back_output

    @lazy_property
    def loss(self):
        cross_entropy = -tf.reduce_sum(self.labels * tf.log(self.train), axis=2)
        mask =tf.sign(tf.reduce_max(tf.abs(self.labels), axis=2))
        reduced_loss = mask * cross_entropy 
        # 计算每个batch的loss
        loss = tf.reduce_sum(reduced_loss, axis=0)
        # reduce_mean后面的数据应该是每个batch的total loss
        loss_final = tf.reduce_mean(loss / tf.cast(self._seq_length, tf.float32))
        return loss_final


    @lazy_property
    def optimize(self):
        optimizer = tf.train.MomentumOptimizer(
            #learning_rate = self.params.learning_rate,
            learning_rate = self.learning_rate,
            momentum = self.params.momentum,
        )

        gradients = optimizer.compute_gradients(self.loss)
        if self.params.gradient_clipping:
            limit = self.params.gradient_clipping
            gradients = [
                (tf.clip_by_value(g, -limit, limit), v)
                if g is not None else (None, v)
                for g, v in gradients
            ]
        optimized = optimizer.apply_gradients(gradients, global_step=self.global_step)
        return optimized


    @lazy_property
    def _seq_length(self):
        # calculate seq length
        signed_input = tf.sign(tf.abs(self.inputs))
        length = tf.reduce_sum(tf.reduce_max(signed_input, axis=2), axis=0)
        return tf.cast(length, dtype=tf.int32)

    @lazy_property
    def prediction(self):
        return tf.argmax(self.train, axis=2, name="prediction")

    @lazy_property
    def test(self):
        return tf.reverse_sequence(self.inputs, seq_lengths=tf.cast(self._seq_length, dtype=tf.int32), batch_axis=1, seq_axis=0)


if __name__ == "__main__":
    f_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./data/letter.data")
    params = AttrDict(
        # 这里面没有将hidden_size直接写成26，后面加一层处理
        rnn_hidden_size=64,
        output_classes=26,
        max_word_length=14,
        batch_size=2,
        single_image_length=128,
        file_path=f_path,
        # bidirectional_rnn 主模型采用了expentional_decay
        learning_rate=0.05,
        momentum=0.5,
        # 加了gradient clipping缓解了发散
        gradient_clipping=0.5,
    )

    data_pool = DataPool(
        params.file_path, 
        params.batch_size, 
        params.max_word_length, 
        params.single_image_length,
    )

    model = BidirectionRnnModule(params)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for data, label in next(data_pool):
            

            feed_dict = {
                model.inputs: data,
                model.labels: label,
            }
            output = sess.run([model.test], feed_dict=feed_dict)
            """
            with open('./tmp_input', 'w') as fd:
                #json.dump(data.tolist(), fd)
                json.dump(data, fd)
            
            a = np.array(output)
            with open('./tmp_output', 'w') as fd:
                json.dump(a.tolist(), fd)
            """

            break
