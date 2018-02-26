import tensorflow as tf 

data = [[3, 4, 2, 1, 0, 0],
        [2, 1, 3, 4, 0, 0],
        [3, 1, 0, 0, 4, 2]]

# batch_size * sequence_length

input_data = tf.placeholder(shape=(None, 6), dtype=tf.int32, name="input")

ops = input_data
with tf.Session() as sess:
    feed_dict = {
        "input:0": data,
    }
    print(sess.run(ops, feed_dict=feed_dict))
