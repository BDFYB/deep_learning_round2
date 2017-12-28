import tensorflow as tf 
from tensorflow.python.framework import graph_util
from tools import data_processor
import train

FROZEN_DIR = './frozen_pb/'
FROZEN_FILE = 'graph.pb'
SUMMARY_DIR = './tensorboard_my_graph/'
DATA_FILENAME = './data/varify.csv'
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
        outputs = graph.get_tensor_by_name('trained/predict:0')
        features, labels = train.make_inputs(BATCH_SIZE)

    with tf.Session(graph=graph) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(10):
            input_feature, survival = sess.run((features, labels))
            feed_dict = {
                inputs: input_feature,
            }
            print(sess.run(outputs, feed_dict))
            print("real:%s" % survival)
        coord.request_stop()
        coord.join(threads)


