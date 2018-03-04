import tensorflow as tf 
from tools.data_pool import DataPool

FROZEN_GRAPH_FILE = "./frozen_graph/graph.pb"
DATA_DIR = './data/aclImdb/test'

def _main():
    # 加载计算图
    # parse the graph_def file

    with tf.gfile.GFile(FROZEN_GRAPH_FILE, "rb") as f:
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
        inputs = graph.get_tensor_by_name('trained/input_data:0')
        loss = graph.get_tensor_by_name('trained/loss:0')
        out_softmax = graph.get_tensor_by_name('trained/softmax_prediction:0')

    # 数据构造
    data_generator = data_pool.DataPool(data_top_dir=DATA_DIR, is_use_embedding=False, batch_size=1)
    with tf.Session() as sess:
        for sentence, label in next(data_generator):
            feed_dict = {
                inputs: sentence,
            }
            predict = sess.run([softmax_prediction], feed_dict=feed_dict)
            print(label)
            print(predict)
            break

if __name__ == "__main__":
    _main()