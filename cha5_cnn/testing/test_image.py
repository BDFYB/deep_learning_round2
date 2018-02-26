import tensorflow as tf
import os

IMAGE = '/home/fengyibo/workspace/deep_learning_round2/cha5_cnn/data/test_data/n02108089-boxer/n02108089_625.jpg'
SUMMARY_DIR = './tensorboard'


def get_image():
    with open(IMAGE, 'rb') as fd:
        image_data = fd.read()
    image = tf.image.decode_jpeg(image_data, channels=3)
    #tf.image.convert_image_dtype 是图片归一化！注意处理顺序
    image_infloat = tf.image.convert_image_dtype(tf.expand_dims(image, 0), tf.float32)
    ori_shape = tf.shape(image)
    return image_infloat, ori_shape


def build_graph_in_nn():
    image, ori_shape = get_image()
    filters = tf.Variable(tf.random_normal([3,3,3,4]))
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
    transpose = tf.layers.conv2d(
        inputs=image,
        filters=4,
        strides=[3, 3],
        kernel_size=(5, 5),
        padding='SAME',
        kernel_initializer=tf.constant_initializer(1),
    )
    shape = tf.shape(transpose)
    return image, ori_shape, shape, transpose

def change_image_to_rgb():
    image, ori_shape = get_image()

    grayscale_image = tf.image.rgb_to_grayscale(image)
    resized_image = tf.image.resize_images(grayscale_image, [250, 151])
    shape = tf.shape(resized_image)
    return image, ori_shape, shape, resized_image

def test_change_image_to_rgb():
    with open(IMAGE, 'rb') as fd:
        image_data = fd.read()
        image = tf.image.decode_jpeg(image_data, channels=3)
    ori_shape = tf.shape(image)
    #这里可以做一些图片处理，翻转、剪裁等
    #这里图片尺寸不一致，这里进行尺寸统一+灰度处理
    try:
        grayscale_image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize_images(grayscale_image, [250, 151])
    except Exception as e:
        print("rgb and resize %s failed: %s" % (file_name, e))
    image_ret = tf.image.convert_image_dtype(image, tf.float32)
    shape = tf.shape(image_ret)
    return tf.expand_dims(image, 0), ori_shape, shape, tf.expand_dims(image_ret, 0)

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
    #image, ori_shape, shape, transformed = test_change_image_to_rgb()
    image, ori_shape, shape, transformed = change_image_to_rgb()
    tf.summary.image("image1", image)
    tf.summary.image("image2", transformed)
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        image_mat, shape1, shape2, resized_image, summ = \
                    sess.run([image, ori_shape, shape, transformed, summary_op])
        print(image_mat)
        print(shape1)
        print(shape2)
        summary_writer.add_summary(summ)
        summary_writer.close()



if __name__ == "__main__":
    if os.path.isdir(SUMMARY_DIR):
        os.system("rm -rf %s" % SUMMARY_DIR)
    _main()
