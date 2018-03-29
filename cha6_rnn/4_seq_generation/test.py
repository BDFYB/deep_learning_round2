import tensorflow as tf

inputs = [[[0, 1, 0, 0], [1, 0, 0, 0]], 
          [[1, 0, 0, 0], [0, 1, 0, 0]], 
          [[0, 0, 1, 0], [0, 0, 1, 0]], 
          [[0, 0, 0, 1], [0, 0, 0, 0]], 
          [[0, 0, 0, 0], [0, 0, 0, 0]]]

with tf.Session() as sess:
    data = tf.transpose(tf.argmax(inputs, axis=2), perm=[1,0])
    print(sess.run(data))
    