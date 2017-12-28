import tensorflow as tf 
import numpy as np 
import os

SUMMARY_DIR = './tensorboard_my_graph'

def get_graph(graph):
    with graph.as_default():
        input_data = tf.placeholder("int32", shape=(2, 2))

        dat_1 = tf.multiply(input_data, input_data)
        var1 = tf.Variable(1)
        var = var1.assign(var1*2)
        ret = tf.add(dat_1, var)

    return input_data, var, ret

    
def _main():
    curr_graph = tf.Graph()
    input_data, var1, ret = get_graph(curr_graph)
    with curr_graph.as_default():
        tf.summary.histogram('all', ret)
        tf.summary.scalar('ret', tf.reduce_mean(ret))
        tf.summary.scalar('var', var1)
        merged_summary = tf.summary.merge_all()

    with tf.Session(graph = curr_graph) as sess:
        train_summary = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for count in range(5):
            
            _, sumerize = sess.run([ret, merged_summary], feed_dict = {
                input_data:[[1, 1], [2, 2]],
            })
            train_summary.add_summary(sumerize, count)
        train_summary.close()
        # 写入上下文管理器不需要手动close
        #sess.close()


if __name__ == '__main__':
    if os.path.isdir(SUMMARY_DIR):
        os.system("rm -rf %s" % SUMMARY_DIR)
    _main()
