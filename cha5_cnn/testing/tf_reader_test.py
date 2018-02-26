import tensorflow as tf
import os
import sys
import json

LABEL_MAP_RELATIVE_PATH = '../data/label_map'
TFRECORDS_RELATIVE_DIR = '../data/tf_records/train_data/'
SUMMARY_DIR = './tensorboard'
BATCH_SIZE = 1

def get_tfrecords_file_list(file_abspath):
    real_subfile_list = []
    subfiles = os.listdir(file_abspath)
    for file in subfiles:
        if ".tfrecords" in file:
            dir_path = os.path.join(file_abspath, file)
            real_subfile_list.append(dir_path)

    return real_subfile_list


def read_label_maps(file_path):
    """
    读取持久化的label_map，make_label_maps调用一次后，均用read_label_maps读取存储结果
    """
    with open(file_path, "r") as fd:
        label_map = json.load(fd)

    #print("read label map:")
    #print(label_map)
    return label_map

def build_graph(file_list, batch_size):
    file_queue = tf.train.string_input_producer(file_list)
    reader = tf.TFRecordReader()
    _, serailized_data = reader.read(file_queue)
    tf_record_features = tf.parse_single_example(serailized_data,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
        }
    )
    tf_record_features = batched_data = tf.train.shuffle_batch(tf_record_features,
        batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size)

    #注意：decode_raw的格式一定要注意！要跟生成时的格式一样。
    #否则会出现数据不一致的情况，比如tf.uint8时，会多出4倍的数据
    tf_record_image = tf.decode_raw(tf_record_features['image'], tf.float32)
    image = tf.reshape(tf_record_image, [BATCH_SIZE, 250, 151, 1])
    tf_records_label = tf_record_features['label']
    return tf_records_label, tf_record_image, image

def _main(label_map, label, tf_record_image, image):
    image_summary_op = tf.summary.image("image", image)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(5):
            runned_label, record_image, runned_image, sums = sess.run((label, tf_record_image, image, image_summary_op))
            print(runned_label.shape)
            print(runned_label)
            print(runned_image)
            """
            print(runned_label)
            print(record_image.shape)
            print(runned_image.shape)
            for single_label in runned_label:
                print(list(label_map.keys())[list(label_map.values()).index(single_label)])
            """
            summary_writer.add_summary(sums)
        coord.request_stop()
        coord.join(threads)
        summary_writer.close()

if __name__ == "__main__":
    if os.path.isdir(SUMMARY_DIR):
        os.system("rm -rf %s" % SUMMARY_DIR)

    file_path = os.path.split(os.path.realpath(__file__))[0]
    record_path = os.path.join(file_path, TFRECORDS_RELATIVE_DIR)
    tf_records_file_list = get_tfrecords_file_list(record_path)
    #print(tf_records_file_list)

    label_map_path = os.path.join(file_path, LABEL_MAP_RELATIVE_PATH)
    label_map = read_label_maps(label_map_path)

    label, tf_record_image, image = build_graph(tf_records_file_list, BATCH_SIZE)
    _main(label_map, label, tf_record_image, image)



