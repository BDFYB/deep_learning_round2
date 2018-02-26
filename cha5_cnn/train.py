import tensorflow as tf
from tensorflow.python.framework import graph_util
import model
import json
import os

SUMMARY_DIR = './summary'
FORZEN_GRAPH_DIR = './frozen_graph/'
FROZEN_FILE = "graph.pb"
TRAIN_DATA_DIR = './data/tf_records/test_data'
BATCH_SIZE = 20
LOOP_TIME = 1000

def push_data_to_graph(tfdata_abs_dir):
    real_subfile_list = []
    subfiles = os.listdir(tfdata_abs_dir)
    for file in subfiles:
        if ".tfrecords" in file:
            dir_path = os.path.join(tfdata_abs_dir, file)
            real_subfile_list.append(dir_path)

    file_queue = tf.train.string_input_producer(real_subfile_list)
    reader = tf.TFRecordReader()
    _, serailized_data = reader.read(file_queue)
    tf_record_features = tf.parse_single_example(serailized_data,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
        }
    )
    batched_tf_record_features = tf.train.shuffle_batch(tf_record_features,\
        batch_size=BATCH_SIZE, capacity=BATCH_SIZE * 50, min_after_dequeue=BATCH_SIZE)
    tf_record_image = tf.decode_raw(batched_tf_record_features['image'], tf.float32)
    image = tf.reshape(tf_record_image, [BATCH_SIZE, 250, 151, 1])
    label = batched_tf_record_features['label']
    return image, label

def train():
    current_file_path = os.path.split(os.path.realpath(__file__))[0]
    tf_data_dir = os.path.join(current_file_path, TRAIN_DATA_DIR)
    image, label = push_data_to_graph(tf_data_dir)
    train_model = model.ImageCnnModel()

    tf.summary.scalar("train_loss", train_model.loss)
    tf.summary.scalar("learning_rate", train_model.learning_rate)
    summarys = tf.summary.merge_all()

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_node_names = ['max_pool_one', 'max_pool_two','softmax', 'input_image', 'loss', 'inference']

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, graph=graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(LOOP_TIME):
            if i % 500 == 0:
                print("current step: %s" % i)
                summary_writer.flush()
            input_image, input_label = sess.run((image, label))
            #print(input_image)
            feed_dict = {
                train_model.image: input_image,
                train_model.label: input_label,
            }

            _, run_summary = sess.run([train_model.optimize, summarys], feed_dict=feed_dict)
            summary_writer.add_summary(run_summary, i)

        coord.request_stop()
        coord.join(threads)

        # save pb file
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names, # We split on comma for convenience
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(FORZEN_GRAPH_DIR + FROZEN_FILE, "wb") as f:
            f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

        summary_writer.close()

if __name__ == "__main__":
    if os.path.isdir(SUMMARY_DIR):
        os.system("rm -rf %s" % SUMMARY_DIR)
    if os.path.isdir(FORZEN_GRAPH_DIR):
        os.system("rm -rf %s"%FORZEN_GRAPH_DIR)
    os.makedirs(FORZEN_GRAPH_DIR)

    train()
