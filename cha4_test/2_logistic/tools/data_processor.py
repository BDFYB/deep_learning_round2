import tensorflow as tf 
import os

def read_cvs(file_name, batch_size, default_records):
    file_name_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])
    reader = tf.TextLineReader(skip_header_lines=1)
    name, value = reader.read(file_name_queue)
    decoded = tf.decode_csv(value, record_defaults=default_records)
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size)

if __name__ == "__main__":
    file_name = "../data/train.csv"
    #test.csv少数据，survive数据。
    default_csv_data = [[0.0], [0.0], [0.0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]]
    batch_data = read_cvs(file_name, 1, default_csv_data)
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = batch_data

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(10):
            sess.run(passenger_id)
            
        coord.request_stop()
        coord.join(threads)