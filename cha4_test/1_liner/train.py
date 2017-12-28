import model
import tensorflow as tf 
from tensorflow.python.framework import graph_util
import numpy as np
import os
import time


FROZEN_DIR = './frozen_pb/'
FROZEN_FILE = 'graph.pb'
SUMMARY_DIR = './tensorboard_my_graph/'

def make_data():
    weight_age = np.array([[84, 46], [73, 20], [65, 52], [70, 30], \
                           [76, 57], [69, 25], [63, 28], [72, 36], \
                           [79, 57], [75, 44], [27, 24], [89, 31], \
                           [65, 52], [57, 23], [59, 60], [69, 48], \
                           [60, 34], [79, 51], [75, 50], [82, 34], \
                           [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]])
    blood_presure = np.array([354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, \
                              290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244])
    length = len(blood_presure)
    return weight_age, blood_presure.reshape(length, 1), length

def _main():
    train_model = model.LinerRegresion()
    inputs, label, data_length = make_data()

    tf.summary.scalar('loss', train_model.loss)
    loss_sum = tf.summary.merge_all()

    graph = tf.get_default_graph()  
    input_graph_def = graph.as_graph_def()  
    output_node_names = ['weights', 'bias', 'input']

    with tf.Session() as sess:
        train_summary = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        for index in range(10000):
            feed_dict = {
                train_model.input: inputs,
                train_model.label: label, 
            }
            _, total_loss, summarys = sess.run([train_model.optimize, train_model.loss, loss_sum], feed_dict)

            train_summary.add_summary(summarys, index)

        train_summary.close()

        # save pb file
        output_graph_def = graph_util.convert_variables_to_constants(  
            sess,   
            input_graph_def,   
            output_node_names, # We split on comma for convenience  
        )   

        # Finally we serialize and dump the output graph to the filesystem  
        with tf.gfile.GFile(FROZEN_DIR + FROZEN_FILE, "wb") as f:  
            f.write(output_graph_def.SerializeToString())  
            print("%d ops in the final graph." % len(output_graph_def.node))     


if __name__ == "__main__":
    if os.path.isdir(SUMMARY_DIR):
        os.system("rm -rf %s" % SUMMARY_DIR)

    if os.path.isdir(FROZEN_DIR):
        os.system("rm -rf %s" % FROZEN_DIR)
    os.makedirs(FROZEN_DIR)

    _main()





