import tensorflow as tf 
import os
import numpy as np
from tools.data_pool import DataPool
from tools.lazy_property import lazy_property
from tools.attr_dict import AttrDict

DATA_REAL_PATH = "./data/arxiv.txt"

class RnnModel(object):

    def __init__(self, params):
        # rnn动力学里面的dtype要与这个inputs的dtype保持一致，否则会报错
        self.inputs = tf.placeholder(shape=(params.seq_length, None, params.vocab_size), dtype=tf.float32, name="input")
        self.label = self._get_label
        self.rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=params.rnn_hidden)
        self.params = params

        self.train
        self.optimize
        self.infer

    @lazy_property
    def train(self):

        rnn_output, state = tf.nn.dynamic_rnn(
            cell=self.rnn_cell,
            inputs=self.inputs,
            sequence_length=self._actual_length,
            time_major=True,
            dtype=tf.float32)

        reshaped_output = tf.reshape(rnn_output, (-1, self.params.rnn_hidden))

        fully_connnected_layer = tf.contrib.layers.fully_connected(
            inputs=reshaped_output,
            num_outputs=self.params.output_size,
            #activation_fn=tf.nn.relu,
            weights_initializer=tf.constant_initializer(1),
            biases_initializer=tf.constant_initializer(0.3),
        )

        softmax = tf.nn.softmax(fully_connnected_layer)

        output = tf.reshape(softmax, (self.params.seq_length, -1, self.params.output_size))

        # ret: seq_length * batch_size * output_size
        return output

    @lazy_property
    def loss(self):
        #将输出的最后一个时间步去掉，match label的shape
        #在计算loss的 mask的时候长度需要减一
        trained_cut = tf.slice(self.train, begin=(0, 0, 0), size=(self.params.seq_length-1, -1, -1))
        mask = tf.reduce_max(self.label, axis=2)
        cross_entropy = -tf.reduce_sum(self.label * tf.log(trained_cut), axis=2)
        loss_total = tf.reduce_sum(cross_entropy * mask, axis=0)
        loss = tf.reduce_mean(loss_total/tf.cast(self._actual_length, tf.float32))
        return loss


    @lazy_property
    def optimize(self):
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=self.params.learning_rate,
            momentum=self.params.momentum)
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
    def _actual_length(self):
        reduced = tf.reduce_max(tf.sign(tf.abs(self.inputs)), axis=2)
        length = tf.reduce_sum(reduced, axis=0)
        return length

    @lazy_property
    def _get_label(self):
        #根据input制作label，抽取从第二个时间点到最后的数据（除去起始数据）
        return tf.slice(self.inputs, begin=(1, 0, 0), size=(-1, -1, -1))

    @lazy_property
    def infer(self):
        max_list = tf.argmax(self.train, axis=2)
        predict = tf.transpose(max_list, perm=[1,0], name="infer")
        return predict
        

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_REAL_PATH)
    params = AttrDict(
        batch_size=2,
        seq_length=1140,
        vocab_size=86,
        rnn_hidden=86,
        output_size=86,
        learning_rate=0.1,
        momentum=0.5,
        gradient_clipping=0.5
    )

    data_pool = DataPool(data_path, params.batch_size, params.seq_length)
    model = RnnModel(params)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index, text_in_onehot in enumerate(next(data_pool)):
            #print(np.array(text_in_onehot).shape)
            feed_dict = {
                model.inputs: text_in_onehot,
            }
            print(sess.run(model.infer, feed_dict=feed_dict))
            break
