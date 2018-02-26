import tensorflow as tf 
from tools.lazy_property import lazy_property


class EmbeddingModel(object):

    def __init__(self, embedding_size, vocab_size, sample_num, momentum, learning_rate):
        self.input_word_ids = tf.placeholder(tf.int32, shape=(None), name="input_word_ids")
        self.output_word_list = tf.placeholder(tf.int32, shape=(None), name="output_word_list")
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.sample_num = sample_num

        init = tf.random_uniform(
            shape=(self.vocab_size, self.embedding_size),
            minval=-0.5,
            maxval=0.5,
            dtype=tf.float32
        )
        self.embedding_map = tf.Variable(
            initial_value=init,
            name="embedding_map",
        )

        self.loss
        self.optimize

    @lazy_property
    def loss(self):
        #embedded = tf.mat_mul(self.input_word_list, self.embedding_map)
        #这里直接用embedding_lookup就行
        embedded = tf.nn.embedding_lookup(
            params=self.embedding_map,
            ids=self.input_word_ids,
        )
        weights_init = tf.random_uniform(
            shape=(self.vocab_size, self.embedding_size),
            minval=-0.5,
            maxval=0.5,
            dtype=tf.float32
        )
        bias_init = tf.zeros(
            shape=(self.vocab_size),
        )

        weights = tf.Variable(
            initial_value=weights_init,
        )
        bias = tf.Variable(
            initial_value=bias_init,
        )
        output_word_list = tf.expand_dims(self.output_word_list, 1)
        nce_updata = tf.nn.nce_loss(
            weights=weights,
            biases=bias,
            labels=output_word_list,
            inputs=embedded,
            num_sampled=self.sample_num, 
            num_classes=self.vocab_size,
        )
        loss = tf.reduce_mean(nce_updata, name="loss")
        return loss

    @lazy_property
    def optimize(self):
        opt = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
        )
        return opt.minimize(self.loss)

