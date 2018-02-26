import tensorflow as tf 
import os
import json
from model import EmbeddingModel
from tools import data_pool


PAGE_DATA_DIR = "./data/pages.bz2"
VOCAL_DATA_DIR = "./data/vocabulary.bz2"

SUMMARY_DIR = "./summary"
FORZEN_GRAPH_DIR = "./frozen_graph"
FROZEN_FILE = "graph.pb"
RESULT_DATA = "embedding_map"

def _main():
    params = {
        #数据配置
        "vocab_dir": VOCAL_DATA_DIR, 
        "page_data_dir": PAGE_DATA_DIR,
        "batch_size": 5,
        "max_context": 2,
        #模型配置
        "embedding_size": 128, 
        "vocab_size": 10000, 
        "sample_num": 10,
        "momentum": 0.5,
        "learning_rate": 0.01,
    }

    batch_data_generator = data_pool.DataPool(
        params["batch_size"], 
        params["max_context"],
        params["vocab_dir"],
        params["page_data_dir"],
    )
    params["vocab_size"] = batch_data_generator.vocab_size

    embedding_model = EmbeddingModel(
        params["embedding_size"], 
        params["vocab_size"], 
        params["sample_num"], 
        params["momentum"], 
        params["learning_rate"],
    )

    tf.summary.scalar("loss", embedding_model.loss)
    tf.summary.histogram("embedding_map", embedding_model.embedding_map)
    summarys = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        counts = 0
        for inputs, labels in next(batch_data_generator):
            counts += 1

            feed_dict = {
                embedding_model.input_word_ids: inputs,
                embedding_model.output_word_list: labels,
            }

            _, sums = sess.run([embedding_model.optimize, summarys], feed_dict=feed_dict)
            summary_writer.add_summary(sums, counts)
            transformer = sess.run(embedding_model.embedding_map)

            if counts % 5000 == 0:
                print("current steps: %s" % counts)
                summary_writer.flush()
                with open(RESULT_DATA, 'w') as fd:
                    json.dump(transformer.tolist(), fd)


if __name__ == "__main__":

    if os.path.isdir(SUMMARY_DIR):
        os.system("rm -rf %s" % SUMMARY_DIR)
    if os.path.isdir(FORZEN_GRAPH_DIR):
        os.system("rm -rf %s"%FORZEN_GRAPH_DIR)
    os.makedirs(FORZEN_GRAPH_DIR)

    _main()
