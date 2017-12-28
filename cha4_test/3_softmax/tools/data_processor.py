import tensorflow as tf 
import os


def load_batched_data(batch_size, file_name):
    default_records = [[0.0], [0.0], [0.0], [0.0], [""]]
    file_name_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])
    reader = tf.TextLineReader()
    name, value = reader.read(file_name_queue)
    data = tf.decode_csv(value, record_defaults=default_records)
    batched_data = tf.train.shuffle_batch(data, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size)
    f1, f2, f3, f4, label = batched_data
    features = tf.stack([f1, f2, f3, f4])
    features = tf.transpose(features)
    label = tf.stack([tf.equal(label, "Iris-setosa"), tf.equal(label, "Iris-versicolor"), tf.equal(label, "Iris-virginica")])
    label = tf.to_float(tf.transpose(label))
    return features, label


if __name__ == "__main__":
    batch_size = 5
    train_file_name = '../data/iris_train.csv'
    test_file_name = '../data/iris_test.csv'
    inputs, labels = load_batched_data(batch_size, train_file_name)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        fet, la = sess.run((inputs, labels))
        print(fet)
        print(la)
        print(type(la))
        #print(LABEL_MAP[la[0]])

        coord.request_stop()
        coord.join(threads)