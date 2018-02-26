import tensorflow
from tools.lazy_property import lazy_property

"""
本model将embedding训练过程联合到训练过程中，即input未经过embedding处理
"""
class RnnTrainModel(object):
    def __init__(self, sentence_max_length, vocab_size, embedding_size, output_num_units, batch_size, momentum, gradient_clipping=None):
        self.input_data = tf.place_holder(shape=(sentence_length, None), name="input_data")
        self.label = tf.place_holder(shape=(None, 1), name="output_data")
        self.rnn_cell = tf.nn.rnn_cell.GRURNNCell(output_num_units)
        self.batch_size = batch_size
        self.momentum = momentum
        self.gradient_clipping = gradient_clipping

        init_random = tf.random_uniform(
            shape=(vocab_size, embedding_size),
            minval=-1,
            maxval=1,
            dtype=tf.float32,
        )
        self.embedding_map = tf.variable(
            initial_value=init_random,
            name="embedding_map",
        )

    @lazy_property
    def train(self):
        # sentence_length * batch_size * embedding_size
        layer_one = tf.nn.embedding_lookup(self.embedding_map, self.input_data)

        output, state = tf.nn.dynamic_rnn(self.cell, layer_one, sequence_length=self.actual_length, time_major=True, dtype=tf.float32)
        # 截取output的每个最后一个输出做全连接（想想有没有必要做，正常RNN输出后取softmax应该可以了）
        # batch_size * output_num_units
        last_output = _last_output(output)

        layer_two = tf.contrib.layers.fully_connected(
            inputs=last_output,
            num_outputs=[self.batch_size, 2],
        )
        return layer_two

    @lazy_property
    def loss(self):
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=self.label,
            logits=self.train,
            name="loss",
        )
        return loss

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
        optimizer = self.params.optimizer.apply_gradients(gradient)
        return optimizer

    @lazy_property
    def infer(self):
        trained = self.train
        prediction = tf.nn.softmax(trained, name="softmax_prediction")
        return prediction

    @lazy_property
    def actual_length(self):
        pass

    @lazy_property
    def _last_output(output):
        pass
    


