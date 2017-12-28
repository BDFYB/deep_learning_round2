import tensorflow as tf 
import numpy as np 
import os

SUMMARY_DIR = './tensorboard_my_graph'


    
def _main():

    with tf.Session() as sess:
        index = [1, 2]
        index2 = tf.Variable(1)
        print(sess.run((tf.shape(index), tf.shape([index2]))))



if __name__ == '__main__':
    if os.path.isdir(SUMMARY_DIR):
        os.system("rm -rf %s" % SUMMARY_DIR)
    _main()