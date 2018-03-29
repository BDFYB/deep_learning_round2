import tensorflow as tf 
from tools.data_pool import DataPool
import os

DATA_REAL_PATH = "./data/arxiv.txt"
FROZEN_GRAPH_FILE = "./frozen/graph.pb"

def _main(params, data_path):
    with tf.gfile.GFile(FROZEN_GRAPH_FILE, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

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
        inputs = graph.get_tensor_by_name('trained/input:0')
        predict = graph.get_tensor_by_name('trained/infer:0')

    infer_data_pool = DataPool(data_path, params.batch_size, params.seq_length)

    with tf.Session(graph=graph) as sess:
        for steps, text_inputs in enumerate(next(infer_data_pool)):
            feed_dict={
                'trained/input:0': text_inputs,
            }

            ret = sess.run("trained/infer:0", feed_dict=feed_dict)
            print(ret)
            break



if __name__ == "__main__":

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