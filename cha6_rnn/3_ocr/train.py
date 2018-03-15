#!/bin/python


from tools.data_pool import DataPool
from tools.attr_dictionary import AttrDict 
from rnn_model import RnnModule
from bidirectional_rnn_model import BidirectionRnnModule
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os
import json
import numpy as np

DATA_REAL_PATH = "./data/letter.data"
SUMMARY_DIR = "./summary"

IS_USE_BIDIRECTIONAL_RNN_MODULE = True
if IS_USE_BIDIRECTIONAL_RNN_MODULE:
    FROZEN_GRAPH_FILE = "./frozen_graph/graph.bidirectional.pb"
    print("use bidirectional rnn train")
else:
    FROZEN_GRAPH_FILE = "./frozen_graph/graph.pb"

def _main(params):

    data_pool = DataPool(
        params.file_path, 
        params.batch_size, 
        params.max_word_length, 
        params.single_image_length,
    )

    data_pool_two = DataPool(
        params.file_path, 
        params.batch_size, 
        params.max_word_length, 
        params.single_image_length,
    )

    if IS_USE_BIDIRECTIONAL_RNN_MODULE:
        model = BidirectionRnnModule(params)
        lerning_rate_summary = tf.summary.scalar("train_rate", model.learning_rate)
        summary_op = tf.summary.merge([lerning_rate_summary])
    else:
        model = RnnModule(params)

    loss_summary = tf.summary.scalar("train_loss", model.loss)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        train_summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        sess.run(tf.global_variables_initializer())
        steps = 0

        for data, label in next(data_pool_two):
            steps += 1
            if steps % 50 == 0:
                print("current_step: %s" % steps)
                train_summary_writer.flush()

            feed_dict = {
                model.inputs: data,
                model.labels: label,
            }

            # train mode

            _, summaries = sess.run([model.optimize, summary_op], feed_dict=feed_dict)
            train_summary_writer.add_summary(summaries, steps)

        # 加了两层之后准确率有所提升
        for data, label in next(data_pool):
            steps += 1
            if steps % 50 == 0:
                print("current_step: %s" % steps)
                train_summary_writer.flush()

            feed_dict = {
                model.inputs: data,
                model.labels: label,
            }
            # test mode
            """
            #print("input data shape:"); print(np.array(data).shape)

            output = sess.run([model.loss], feed_dict=feed_dict)
            print(output)
            
            break
            a = np.array(output)
            print(a.shape)
            with open('./tmp_output', 'w') as fd:
                json.dump(a.tolist(), fd)
            break
            """
            # train mode
            _, summaries = sess.run([model.optimize, summary_op], feed_dict=feed_dict)
            train_summary_writer.add_summary(summaries, steps)

        # 制作frozen文件，持久化变量用于infer
        cur_graph = tf.get_default_graph()
        input_graph_def = cur_graph.as_graph_def()
        output_node_names = ["inputs", "prediction"]
        output_graph_def = graph_util.convert_variables_to_constants(  
            sess,   
            input_graph_def,   
            output_node_names, # We split on comma for convenience  
        ) 
        # Finally we serialize and dump the output graph to the filesystem  
        with tf.gfile.GFile(FROZEN_GRAPH_FILE, "wb") as f:  
            f.write(output_graph_def.SerializeToString())  
            print("%d ops in the final graph." % len(output_graph_def.node)) 


if __name__ == "__main__":

    if os.path.isdir(SUMMARY_DIR):
        os.system("rm -rf %s" % SUMMARY_DIR)

    if os.path.isfile(FROZEN_GRAPH_FILE):
        os.system("rm %s" % FROZEN_GRAPH_FILE)

    f_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_REAL_PATH)

    params = AttrDict(
        # 这里面没有将hidden_size直接写成26，后面加一层处理
        rnn_hidden_size=64,
        output_classes=26,
        max_word_length=14,
        batch_size=2,
        single_image_length=128,
        file_path=f_path,
        learning_rate=0.01,
        momentum=0.5,
        # 加了gradient clipping缓解了发散
        gradient_clipping=0.5,
    )
    _main(params)
