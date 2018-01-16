import tensorflow as tf 
from tools.lazy_property import lazy_property
import model 


class train_model(object):
    def __init__(self):
        self.inputs = tf.placeholder('float32', shape=(None, 10), name="inputs")
        self.label = tf.placeholder('float32', shape=(None, 1), name="label")
        #learning_rate过大很容易发散，尤其层数多的时候
        self.learning_rate = 0.001
        #! 终于知道在这里有什么用了！这里会调用prediction(),会构造计算图。如果只是采用延迟初始化的话，会造成
        # use uninitialized value Variable 报错. fully_connected会返回变量，有变量就不能延时初始化
        self.prediction
        self.infer

    @lazy_property
    def prediction(self):
        hidden_connect_layer = tf.contrib.layers.fully_connected(
            inputs = self.inputs,
            num_outputs = 5,
            activation_fn = None,
            weights_initializer = tf.contrib.layers.xavier_initializer(),
            biases_initializer = tf.contrib.layers.xavier_initializer(),
        )
        #使用两层准确率明显上升！！
        hidden_connect_layer_2 = tf.contrib.layers.fully_connected(
            inputs = hidden_connect_layer,
            num_outputs = 1,
            activation_fn = None,
            weights_initializer = tf.contrib.layers.xavier_initializer(),
            biases_initializer = tf.contrib.layers.xavier_initializer(),
        )
        return hidden_connect_layer_2

    @lazy_property
    def loss(self):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.prediction, name="loss"))
        return loss

    @lazy_property
    def optimize(self):
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    @lazy_property
    def infer(self):
        predicted = tf.cast(tf.sigmoid(self.prediction) > 0.5, tf.float32, name="predict")
        return predicted


