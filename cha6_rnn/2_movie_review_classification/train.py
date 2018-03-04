import os
import json
import tensorflow as tf 
from tensorflow.python.framework import graph_util
from tools import data_pool 
from rnn_model import RnnTrainModel

SUMMARY_DIR = "./summary"
FROZEN_GRAPH_FILE = "./frozen_graph/graph.pb"
IMBEDDING_MAP_FILE = "./data/embedding_map.json"
DATA_DIR = './data/aclImdb/train'
PARAMS = {
    "embedding_size": 128,
    "vocab_size": 89527,
    "sentence_max_length": 2200,
    "output_num_units": 2,
    "batch_size": 2,
    "momentum": 0.5,
    "gradient_clipping": False
}

def main():
    #prepare data
    data_generator = data_pool.DataPool(data_top_dir=DATA_DIR, is_use_embedding=False, batch_size=PARAMS["batch_size"])
    train_model = RnnTrainModel(
        sentence_max_length=PARAMS["sentence_max_length"],
        vocab_size=PARAMS["vocab_size"], 
        embedding_size=PARAMS["embedding_size"], 
        output_num_units=PARAMS["output_num_units"], 
        batch_size=PARAMS["batch_size"], 
        momentum=PARAMS["momentum"], 
        gradient_clipping=None
    )

    loss_summary = tf.summary.scalar("train_loss", train_model.loss)
    summary_op = tf.summary.merge([loss_summary])

    with tf.Session() as sess:
        step_count = 0
        sess.run(tf.global_variables_initializer())
        train_summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

        for sentence, label in next(data_generator):
            step_count += 1
            if step_count % 50 == 0:
                print("current_step: %s" % step_count)
                train_summary_writer.flush()

            feed_dict = {
                train_model.input_data: sentence,
                train_model.label: label,
            }
            """下面几行是用来测试获取最后一个RNN输出的数据的
            actual_length, output, output2, gather1, gather2 = sess.run(train_model.test_last_output, feed_dict=feed_dict)
            with open("test", 'w') as fd:
                json.dump(output.tolist(), fd)
            with open("test2", 'w') as fd:
                json.dump(output2.tolist(), fd)
            print(actual_length)
            print(gather1)
            print(gather2)
            exit()
            """
            _, summarys = sess.run([train_model.optimize, summary_op], feed_dict=feed_dict)
            train_summary_writer.add_summary(summarys, step_count)

        # 制作frozen文件，持久化变量用于infer
        cur_graph = tf.get_default_graph()
        input_graph_def = cur_graph.as_graph_def()
        output_node_names = ["softmax_prediction", "input_data", "loss"]
        output_graph_def = graph_util.convert_variables_to_constants(  
            sess,   
            input_graph_def,   
            output_node_names, # We split on comma for convenience  
        ) 
        # Finally we serialize and dump the output graph to the filesystem  
        with tf.gfile.GFile(FROZEN_GRAPH_FILE, "wb") as f:  
            f.write(output_graph_def.SerializeToString())  
            print("%d ops in the final graph." % len(output_graph_def.node)) 
        
        # 训练副产品：embedding map
        embedding_map = sess.run(train_model.embedding_map)
        with open(IMBEDDING_MAP_FILE, "w") as file:
            json.dump(embedding_map.tolist(), file)


if __name__ == "__main__":
    if os.path.isdir(SUMMARY_DIR):
        os.system("rm -rf %s" % SUMMARY_DIR)

    if os.path.isfile(FROZEN_GRAPH_FILE):
        os.system("rm %s" % FROZEN_GRAPH_FILE)

    if os.path.isfile(IMBEDDING_MAP_FILE):
        os.system("rm %s" % IMBEDDING_MAP_FILE)

    if not os.path.isdir(os.path.dirname(FROZEN_GRAPH_FILE)):
        os.mkdir(os.path.dirname(FROZEN_GRAPH_FILE))

    main()
