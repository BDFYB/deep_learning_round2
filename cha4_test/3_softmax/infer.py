import tensorflow as tf 
from tensorflow.python.framework import graph_util
from tools import data_processor


FROZEN_DIR = "./frozen_graph"
FROZEN_FILE = 'graph.pb'

TRAIN_FILE = "./data/iris_test.csv"
BATCH_SIZE = 1


if __name__ == "__main__":
    # 直接加载计算图
    # parse the graph_def file

    with tf.gfile.GFile(FROZEN_DIR + FROZEN_FILE, "rb") as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read()) 

    # load the graph_def in the default graph

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(  
            graph_def,   
            input_map = None,   
            return_elements = None,   
            name = "trained",   
            op_dict = None,   
            producer_op_list = None  
        )
        inputs = graph.get_tensor_by_name('trained/inputs:0')
        softmax = graph.get_tensor_by_name('trained/softmax:0')
        outputs = graph.get_tensor_by_name('trained/infer:0')
        inputs_in_batch, labels_in_batch = data_processor.load_batched_data(BATCH_SIZE, TRAIN_FILE)

    with tf.Session(graph=graph) as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
        for i in range(10):
            input_data, labels = sess.run([inputs_in_batch, labels_in_batch])

            feed_dict = {
                inputs: input_data,
            }
            res_softmax, result = sess.run([softmax, outputs], feed_dict)
            print("real labels: %s" % labels)
            print("prediction softmax: %s " % res_softmax)
            print("prediction: %s " % result)
        coord.request_stop()
        coord.join(threads)
