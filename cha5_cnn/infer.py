import tensorflow as tf 
import json
import os

FORZEN_GRAPH_DIR = './frozen_graph/'
FROZEN_FILE = "graph.pb"
LABEL_MAP_FILE = './data/label_map'
TEST_DATA_DIR = './data/tf_records/train_data'
TF_VERIFY_SUMMARY_DIR = './infer_summary'
BATCH_SIZE = 1


def get_label_map():
    with open(LABEL_MAP_FILE, 'r') as fd:
        label_map = json.load(fd)
    return map(lambda x:(x[1], x[0]) , label_map.items())

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
    #tf_record_features = tf.train.shuffle_batch(tf_record_features,\
    #        batch_size=BATCH_SIZE, capacity=BATCH_SIZE * 50, min_after_dequeue=BATCH_SIZE)
    tf_record_image = tf.decode_raw(tf_record_features['image'], tf.float32)
    image = tf.reshape(tf_record_image, [1, 250, 151, 1])
    label = tf_record_features['label']
    return image, label

def inference():
    label_map = dict(get_label_map())
    #print(label_map)

    # 加载计算图
    # parse the graph_def file

    with tf.gfile.GFile(FORZEN_GRAPH_DIR + FROZEN_FILE, "rb") as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read()) 

    # load the graph_def in the default graph

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(  
            graph_def,   
            input_map = None,   
            return_elements = None,   
            name = "trained",   
            op_dict = None,   
            producer_op_list = None  
        )
        inputs = graph.get_tensor_by_name('trained/input_image:0')
        outputs = graph.get_tensor_by_name('trained/inference:0')
        DATA_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], TEST_DATA_DIR)
        inputs_image, labels = push_data_to_graph(DATA_DIR)
        tf_image = tf.summary.image("image", inputs_image)

    with tf.Session(graph=graph) as sess:
        image_recorder = tf.summary.FileWriter(TF_VERIFY_SUMMARY_DIR)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(20):
            image_sum, image, label = sess.run([tf_image, inputs_image, labels])
            
            feed_dict = {
                inputs: image,
            }
            inference = sess.run(outputs, feed_dict=feed_dict)[0]
            #print("start inference")
            print("ori vs infer: %s:%s" % (label, inference + 1))
            #print("ori_class: %s" % label_map[label])
            #print("inference_class: %s" % label_map[inference+1])
            image_recorder.add_summary(image_sum)

        coord.request_stop()
        coord.join(threads) 


if __name__ == "__main__":
    inference()