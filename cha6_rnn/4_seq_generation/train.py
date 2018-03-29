import tensorflow as tf 
from tensorflow.python.framework import graph_util
import os
import numpy as np
from tools.data_pool import DataPool
from tools.lazy_property import lazy_property
from tools.attr_dict import AttrDict
from rnn_model import RnnModel

DATA_REAL_PATH = "./data/arxiv.txt"
SUMMARY_DIR = "./summary"
FROZEN_GRAPH_FILE = "./frozen/graph.pb"

def _main(params, data_path):
    
    data_pool = DataPool(data_path, params.batch_size, params.seq_length)
    model = RnnModel(params)

    loss_summary = tf.summary.scalar("train_loss", model.loss)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        sess.run(tf.global_variables_initializer())

        for steps, text_inputs in enumerate(next(data_pool)):
            feed_dict = {
                model.inputs: text_inputs,
            }

            if steps % 100 == 0:
                print("current_step: %s" % steps)
                summary_writer.flush()

            _, summaries = sess.run([model.optimize, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summaries, steps)

        # 制作frozen文件，持久化变量用于infer
        cur_graph = tf.get_default_graph()
        input_graph_def = cur_graph.as_graph_def()
        output_node_names = ["input", "infer"]
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

    _main(params, data_path)