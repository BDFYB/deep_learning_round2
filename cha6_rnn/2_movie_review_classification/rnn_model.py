import tensorflow as tf
from tools.lazy_property import lazy_property

"""
本model将embedding训练过程联合到训练过程中，即input未经过embedding处理
"""
class RnnTrainModel(object):
    def __init__(self, sentence_max_length, vocab_size, embedding_size, output_num_units, batch_size, momentum, gradient_clipping=None):
        self.input_data = tf.placeholder(shape=(sentence_max_length, None), name="input_data", dtype=tf.int32)
        self.label = tf.placeholder(shape=(None), name="output_data", dtype=tf.int32)
        self.rnn_cell = tf.nn.rnn_cell.GRUCell(output_num_units)
        self.batch_size = batch_size
        self.momentum = momentum
        self.gradient_clipping = gradient_clipping

        init_random = tf.random_uniform(
            shape=(vocab_size, embedding_size),
            minval=-1,
            maxval=1,
            dtype=tf.float32,
        )
        self.embedding_map = tf.Variable(
            initial_value=init_random,
            name="embedding_map",
        )

        self.train 
        self.loss 
        self.optimize
        self.infer

    @lazy_property
    def train(self):
        # sentence_length * batch_size * embedding_size，维数扩充了
        layer_one = tf.nn.embedding_lookup(self.embedding_map, self.input_data)

        output, state = tf.nn.dynamic_rnn(self.rnn_cell, layer_one, sequence_length=self.actual_length, time_major=True, dtype=tf.float32)
        # 截取output的每个最后一个输出做全连接（想想有没有必要做，正常RNN输出后取softmax应该可以了）
        # output: sentence_length * batch_size * num_units 
        # state: batch_size * num_units

        # batch_size * output_num_units
        last_output = self._last_output(output)

        layer_two = tf.contrib.layers.fully_connected(
            inputs=last_output,
            num_outputs=2,
        )
        return layer_two

    @lazy_property
    def loss(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.label,
            logits=self.train,
        )
        loss_data = tf.reduce_mean(loss, name="loss")
        return loss_data

    @lazy_property
    def optimize(self):
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=0.01,
            momentum=self.momentum,
        )
        gradient = optimizer.compute_gradients(self.loss) 
        if self.gradient_clipping:
            limit = self.gradient_clipping
            gradient = [
                (tf.clip_by_value(g, -limit, limit), v)
                if g is not None else (None, v)
                for g, v in gradient
            ]
        optimizer = optimizer.apply_gradients(gradient)
        return optimizer

    @lazy_property
    def infer(self):
        trained = self.train
        prediction = tf.nn.softmax(trained, name="softmax_prediction")
        return prediction

    @lazy_property
    def actual_length(self):
        # return example: [24, 32], shape of 2
        transport = tf.sign(tf.abs(self.input_data))
        return tf.cast(tf.reduce_sum(transport, axis=0), dtype=tf.int32)

    def _last_output(self, output):
        # 需要将其转换成batch_size * seq_length * num_outputs形式之后用actual_length来gather
        reversed_output = tf.transpose(output, perm=[1,0,2])

        #batch_size = int(reversed_output.get_shape()[0])
        sentence_length = int(reversed_output.get_shape()[1])
        output_size = int(reversed_output.get_shape()[2])

        flat = tf.reshape(reversed_output, [-1, output_size])
        index = tf.range(0, self.batch_size) * sentence_length + (self.actual_length -1)
        return tf.gather(flat, indices=index)

    """
    @lazy_property
    #下面这个函数是用来测试获取RNN执行后最后一个有效数据的。
    #这里面展示了两种方法，结果是一样的，只不过第二种维数多了一维。
    def test_last_output(self):
        # sentence_length * batch_size * embedding_size，维数扩充了
        layer_one = tf.nn.embedding_lookup(self.embedding_map, self.input_data)

        output, state = tf.nn.dynamic_rnn(self.rnn_cell, layer_one, sequence_length=self.actual_length, time_major=True, dtype=tf.float32)

        # 需要将其转换成batch_size * seq_length * num_outputs形式之后用actual_length来gather
        reversed_output = tf.transpose(output, perm=[1,0,2])

        #batch_size = int(reversed_output.get_shape()[0])
        sentence_length = int(reversed_output.get_shape()[1])
        output_size = int(reversed_output.get_shape()[2])

        flat = tf.reshape(reversed_output, [-1, output_size])
        index = tf.range(0, self.batch_size) * sentence_length + (self.actual_length -1)
        return self.actual_length, output, reversed_output, tf.gather(flat, indices=index), tf.gather(output, indices=(self.actual_length-1))
    """


