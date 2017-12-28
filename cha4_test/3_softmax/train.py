import tensorflow as tf 
from tools import data_processor
from tensorflow.python.framework import graph_util
import model
import os

SUMMARY_DIR = "./tensorboard/"
FROZEN_DIR = "./frozen_graph/"
FROZEN_FILE = 'graph.pb'

TRAIN_FILE = "./data/iris_train.csv"
BATCH_SIZE = 135

def _main():
    inputs_in_batch, labels_in_batch = data_processor.load_batched_data(BATCH_SIZE, TRAIN_FILE)
    train_model = model.SoftMaxModel()

    tf.summary.scalar("loss", train_model.loss)
    merged_summary = tf.summary.merge_all()

    graph = tf.get_default_graph()  
    input_graph_def = graph.as_graph_def()  
    output_node_names = ['inputs', 'softmax', 'infer']

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        train_summary = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for index in range(1000):
            input_data, label = sess.run([inputs_in_batch, labels_in_batch])
            feed_dict = {
                train_model.inputs: input_data,
                train_model.labels: label,
            }
            _, infer, summaries = sess.run([train_model.loss, train_model.infer, merged_summary], feed_dict)
            train_summary.add_summary(summaries, index)
            #print(infer)

        coord.request_stop()
        coord.join(threads)
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
