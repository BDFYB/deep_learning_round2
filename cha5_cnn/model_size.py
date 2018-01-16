import tensorflow as tf 
from tools.lazy_property import lazy_property


class ImageCnnModel(object):

    def __init__(self):
        self.image = tf.placeholder(tf.float32, (None, 250, 151, 1), name="input_image")
        self.label = tf.placeholder(tf.int64, (None), name="input_label")

        #这边这种使用方式是可以的，即将tfread与整个模型直接连接起来，但是目前没有找到好的infer形式
        #self.image = train_image
        #self.label = train_label
        
        self.global_step = tf.Variable(1, trainable=False, name="global_steps")
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=0.01,
            global_step=self.global_step,
            decay_steps=120,
            decay_rate=0.9,
            staircase=True,
            name="learning_rate",
        )
        self.train

    @lazy_property
    def train(self):
        # input size: (batch_size, 250, 151, 1)
        conv_layer_one = tf.contrib.layers.conv2d(    
            inputs=self.image,
            num_outputs=32,
            kernel_size=(5, 5),
            padding='SAME',
            activation_fn=tf.nn.relu,
            stride=[2, 2],
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
        )
        self.conv_layer_one_shape = conv_layer_one.get_shape()
        # conv_layer_one size: (batch_size, 125, 76, 32)
        pool_layer_one = tf.nn.max_pool(
            conv_layer_one,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
        )
        self.pool_layer_one_shape = pool_layer_one.get_shape()
        # pool_layer_one size: (batch_size, 63, 38, 32)
        conv_layer_two = tf.contrib.layers.conv2d(
            inputs=pool_layer_one,
            num_outputs=64,
            kernel_size=(5, 5),
            padding='SAME',
            activation_fn=tf.nn.relu,
            stride=[2, 2],
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
        )
        self.conv_layer_two_shape = conv_layer_two.get_shape()
        # conv_layer_two size: (batch_size, 32, 19, 64)
        pool_layer_two = tf.nn.max_pool(
            conv_layer_two,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
        )
        self.pool_layer_two_shape = pool_layer_two.get_shape()
        # pool_layer_two size: (batch_size, 16, 10, 64)
        batch_size, image_hight, image_width, channel = pool_layer_two.get_shape()
        flattened = tf.reshape(pool_layer_two, [-1, image_width.value * image_hight.value * channel.value])
        fully_connect_layer_three = tf.contrib.layers.fully_connected(
            flattened,
            512,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
        )
        hidden_layer_three = tf.nn.dropout(
            fully_connect_layer_three,
            0.5,
        )

        final_layer = tf.contrib.layers.fully_connected(
            hidden_layer_three,
            120,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
        )
        return conv_layer_one, pool_layer_one, conv_layer_two, pool_layer_two

    """
    @lazy_property
    def loss(self):
        #对于标签是值（不是onehot形式）使用sparse方式
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.train, labels=tf.subtract(self.label, 1))
        loss_data = tf.reduce_mean(loss, name="loss")
        return loss_data

    @lazy_property
    def optimize(self):
        #optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.6)
        optimize = optimizer.minimize(self.loss, global_step = self.global_step)
        return optimize

    @lazy_property
    def infer(self):
        result = tf.nn.softmax(self.train, name="softmax")
        inference = tf.argmax(result, 1, name="inference")
        return inference
    """

