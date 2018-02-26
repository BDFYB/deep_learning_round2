import tensorflow as tf 
from tensorflow.python.framework import graph_util
import model_size
import json
import os

TRAIN_DATA_DIR = './data/tf_records/test_data'
BATCH_SIZE = 4
LOOP_TIME = 1

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
    train_model = model_size.ImageCnnModel()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(LOOP_TIME):
            if i % 500 == 0:
                print("current step: %s" % i)
            input_image, input_label = sess.run((image, label))
            feed_dict = {
                train_model.image: input_image,
                train_model.label: input_label,
            }
            
            print(sess.run([train_model.train], feed_dict=feed_dict))

        coord.request_stop()
        coord.join(threads)  


if __name__ == "__main__":

    train()
