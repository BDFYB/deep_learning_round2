import tensorflow as tf
import os
import sys
import json

LABEL_MAP_RELATIVE_PATH = '../data/label_map'
DATA_RELATIVE_PATH = '../data/train_data'
TFRECORDS_RELATIVE_DIR = '../data/tf_records/train_data/'
TEMP_FILE = './label_saved_order'

def get_dir_subfile_list(data_abspath, is_get_subdir):
    real_subfile_list = []
    subfiles = os.listdir(data_abspath)
    if is_get_subdir:
        for file in subfiles:
            dir_path = os.path.join(data_abspath, file)
            if (os.path.isdir(dir_path)):
                real_subfile_list.append(dir_path)
    else:
        for file in subfiles:
            dir_path = os.path.join(data_abspath, file)
            if (os.path.isfile(dir_path)):
                real_subfile_list.append(dir_path)

    return real_subfile_list

def get_image_data(data_abspath):
    data_map = {}
    subdir_list = get_dir_subfile_list(data_abspath, True)
    for subdir in subdir_list:
        data_map[subdir] = get_dir_subfile_list(subdir, False)
    return data_map

def make_label_maps(data_abspath, data_map):
    """
    读取所有的label，制作成稀疏编码，并返回及持久化编码文件，仅使用一次即可，后面读对应持久化的文件就行
    最后存储到tfrecords里面的label数据为1，2，3，4....
    """
    label_map = {}
    count = 0
    for dirs, jpeg_file_list in data_map.items():
        real_dir = dirs.split('/')[-1]
        if real_dir not in label_map:
            label_map[real_dir] = count
            count += 1
    with open(LABEL_MAP_RELATIVE_PATH, "w") as fd:
        label_injson = json.dump(label_map, fd)

    print("make label map:")
    print(label_map)
    return label_map

def read_label_maps():
    """
    读取持久化的label_map，make_label_maps调用一次后，均用read_label_maps读取存储结果
    """
    with open(LABEL_MAP_RELATIVE_PATH, "r") as fd:
        label_map = json.load(fd)

    print("read label map finish")
    #print(label_map)
    return label_map

def make_tf_records(data_map, label_map):

    file_counts = 0
    tf_records_fd = None
    sess = tf.Session()
    random_data_map = {}
    #记录下存储tf_records的顺序
    saved_label_inorder_record = []

    # 多个tf文件不能保证纯随机获取数据
    # 经实践检验，单个tfrecord依旧是按照大致先后顺序读取的，没有太大的打乱
    # 采用数据存储前就打乱的方式，之后存储多个tf文件的方案
    """
    tfrecords_file_name = "{output_dir}{name}.tfrecords".format(
        output_dir = TFRECORDS_RELATIVE_DIR,
        name = "image",
    )
    tf_records_fd = tf.python_io.TFRecordWriter(tfrecords_file_name)
    """

    #构造随机数据dict
    total_image_count = 0
    for key, file_name_lists in data_map.items():
        for file_name in file_name_lists:
            random_data_map[file_name] = key
            total_image_count += 1

    for file_name, key in random_data_map.items():
        if file_counts % 5000 == 0:
            print("current_step: %s" % file_counts)

            if tf_records_fd:
                tf_records_fd.close()
            tfrecords_file_name = "{output_dir}{count_index}.tfrecords".format(
                output_dir = TFRECORDS_RELATIVE_DIR,
                count_index = file_counts)
            tf_records_fd = tf.python_io.TFRecordWriter(tfrecords_file_name)


        with open(file_name, 'rb') as fd:
            image_data = fd.read()
            image = tf.image.decode_jpeg(image_data, channels=3)
        #tf.image.convert_image_dtype 是图片归一化！一定要在这里处理，先后顺序会有影响
        image = tf.image.convert_image_dtype(image, tf.float32)
        #这里可以做一些图片处理，翻转、剪裁等
        #这里图片尺寸不一致，这里进行尺寸统一+灰度处理
        try:
            grayscale_image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize_images(grayscale_image, [250, 151])
        except Exception as e:
            print("rgb and resize %s failed: %s" % (file_name, e))
            continue


        #千万不要在这里with tf.Session() as sess,这样每步都会创建一个会话并删除
        try:
            #可能读取一些wrong格式的jpg，需要catch
            image_runned = sess.run(image)
        except Exception as e:
            print("sess run jpeg %s failed: %s" % (file_name, e))
            continue

        #image_hight, image_width, image_channel = image_runned.shape
        #print(image_runned.shape)
        image_bytes = image_runned.tobytes()
        real_dir = key.split('/')[-1]
        if real_dir not in label_map:
            print("%s not in label_map!" % real_dir)
            continue
        image_label = label_map[real_dir]
        saved_label_inorder_record.append(image_label)

        example_proto = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[image_label])),
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
        }))
        tf_records_fd.write(example_proto.SerializeToString())
        file_counts += 1

    print("total_image_count:%s"%total_image_count)
    print("file_counts:%s"%file_counts)
    with open(TEMP_FILE, "w") as fd:
        label_injson = json.dump(saved_label_inorder_record, fd)

    tf_records_fd.close()
    sess.close()


if __name__ == "__main__":
    file_path = os.path.split(os.path.realpath(__file__))[0]
    data_path = os.path.join(file_path, DATA_RELATIVE_PATH)

    data_map = get_image_data(data_path)
    # 持久化label_map文件仅需要制作一次
    #label_map = make_label_maps(data_path, data_map)
    label_map = read_label_maps()

    make_tf_records(data_map, label_map)
