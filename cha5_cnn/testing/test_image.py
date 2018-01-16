import tensorflow as tf 
import numpy as np
import os

IMAGE = '/Users/baidu/AI/deep_learning_round2/cha5_cnn/data/test_data/n02108089-boxer/n02108089_625.jpg'
SUMMARY_DIR = './tensorboard'


def get_image():
    with open(IMAGE, 'rb') as fd:
        image_data = fd.read()
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.convert_image_dtype(tf.expand_dims(image, 0), tf.float32)
    ori_shape = tf.shape(image)
    return image, ori_shape


def build_graph_in_nn():
    image, ori_shape = get_image()
    #[filter_height, filter_width, in_channels, out_channels]
    filters = tf.Variable(tf.random_normal([4,3,3,1]))
    transpose = tf.nn.conv2d(    
        input=image,
        filter=filters,
        strides=[1,3,3,1],
        padding='VALID',
    )
    shape = tf.shape(transpose)
    return image, ori_shape, shape, transpose


def build_graph_in_layers():
    image, ori_shape = get_image()
    #shape of image: [1 500 375   3]
    #shape of output: [1 167 125   4]
    transpose = tf.layers.conv2d(    
        inputs=image,
        filters=4,
        strides=[3, 3],
        kernel_size=(5, 5),
        padding='SAME',
        kernel_initializer=tf.constant_initializer(1),
        name='conv1'
    )
    #能够看出卷积核变量的大小为：(1, 5, 5, 3, 4)，即channel通道会自动填充！
    kernel = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/kernel')
    
    shape = tf.shape(transpose)
    return image, ori_shape, shape, transpose, kernel

def change_image_to_rgb():
    image, ori_shape = get_image()
    grayscale_image = tf.image.rgb_to_grayscale(image)
    resized_image = tf.image.resize_images(grayscale_image, [250, 151])
    shape = tf.shape(resized_image)
    return image, ori_shape, shape, resized_image


def build_graph_to_test_kernel():
    image, ori_shape = get_image()
    """
    size:3*3*3*3, 3个卷积核
    """
    kernel_boundary = tf.constant(
        [
            [
                [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
                [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
                [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
            ],
            [
                [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
                [[ 8., 0., 0.], [0.,  8., 0.], [0., 0.,  8.]],
                [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
            ],
            [
                [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
                [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
                [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
            ],
        ]
    )
    kernel_sharp = tf.constant(
        [
            [
                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
            ],
            [
                [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
                [[ 5., 0., 0.], [0.,  5., 0.], [0., 0.,  5.]],
                [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
            ],
            [
                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
            ],
        ]
    )
    filters = tf.Variable(tf.random_normal([3,3,3,4]))
    transpose = tf.nn.conv2d(    
        input=image,
        filter=kernel_sharp,
        strides=[1,1,1,1],
        padding='SAME',
    )
    shape = tf.shape(transpose)
    return image, ori_shape, shape, transpose


def _main():
    #image, ori_shape, shape, transformed = build_graph_in_nn()
    image, ori_shape, shape, transformed, kernel = build_graph_in_layers()
    tf.summary.image("image1", image)
    tf.summary.image("image2", transformed)
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        #image_mat, shape1, shape2, summ = sess.run([image, ori_shape, shape, summary_op])
        #下面这个是打印tf.layers.conv2d的卷积核维数专用
        image_mat, shape1, shape2, summ, kernel = sess.run([image, ori_shape, shape, summary_op, kernel])
        print(np.array(kernel).shape)
        print(shape1)
        print(shape2)
        summary_writer.add_summary(summ)
        summary_writer.close()

    

if __name__ == "__main__":
    if os.path.isdir(SUMMARY_DIR):
        os.system("rm -rf %s" % SUMMARY_DIR)
    _main()
