import tensorflow as tf 
from tools.data_pool import DataPool
from tools.data_pool import LetterMap
import numpy as np

DATA_REAL_PATH = "./data/letter.data"
IS_USE_BIDIRECTIONAL_RNN_MODULE = True
if IS_USE_BIDIRECTIONAL_RNN_MODULE:
    FROZEN_GRAPH_FILE = "./frozen_graph/graph.bidirectional.pb"
    print("use bidirectional rnn infer")
else:
    FROZEN_GRAPH_FILE = "./frozen_graph/graph.pb"

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
        inputs = graph.get_tensor_by_name('trained/inputs:0')
        predict = graph.get_tensor_by_name('trained/prediction:0')

    infer_data_pool =  DataPool(
        data_abs_path=DATA_REAL_PATH, 
        batch_size=1,
        max_word_length=14,
        image_length=128,
    )
    letter_map = LetterMap()

    with tf.Session(graph=graph) as sess:

        test_word_num = 0
        word_right_inference = 0
        test_letter_num = 0
        letter_right_inference = 0
        for data, label in next(infer_data_pool):
            test_word_num += 1
            str_ori_length = 0
            ori_str = ""
            inf_str = ""
            for ori_letter in label:
                if (np.max(ori_letter) == 0):
                    break
                str_ori_length += 1
                ori_str += letter_map.get_letter_by_num(np.argmax(ori_letter[0]))

            feed_dict = {
                inputs: data
            }
            result = sess.run(predict, feed_dict=feed_dict)
            for index in range(str_ori_length):
                inf_str += letter_map.get_letter_by_num(result[index][0])
            if ori_str == inf_str:
                word_right_inference += 1

            for index in range(str_ori_length):
                test_letter_num += 1
                if ori_str[index] == inf_str[index]:
                    letter_right_inference += 1
        print("word accuracy: %s" % float(word_right_inference/test_word_num))
        print("letter accuracy: %s" % float(letter_right_inference/test_letter_num))


if __name__ == "__main__":
    _main()