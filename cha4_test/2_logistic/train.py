import tensorflow as tf 
from tools import data_processor
from tensorflow.python.framework import graph_util
import model
import os


FROZEN_DIR = './frozen_pb/'
FROZEN_FILE = 'graph.pb'
SUMMARY_DIR = './tensorboard_my_graph/'
DATA_FILENAME = './data/train.csv'
BATCH_SIZE = 5

"""
数据集中共有12个字段，PassengerId：乘客编号，Survived：乘客是否存活，Pclass：乘客所在的船舱等级；
Name：乘客姓名，Sex：乘客性别，Age：乘客年龄，SibSp：乘客的兄弟姐妹和配偶数量，Parch：乘客的父母与子女数量，
Ticket：票的编号，Fare：票价，Cabin：座位号，Embarked：乘客登船码头。 
共有891位乘客的数据信息。其中277位乘客的年龄数据缺失，2位乘客的登船码头数据缺失，687位乘客的船舱数据缺失
筛选出有价值的自变量：
Sex、Age、SibSp、Parch、Pclass、Embarked
因变量：
Survived
方法：全连接+sigmoid
"""
def make_inputs(batch_size):
    default_csv_data = [[0.0], [0.0], [2.0], [""], [""], [30.0], [0.0], [0.0], [""], [0.0], [""], ["s"]]
    batched_data = data_processor.read_cvs(DATA_FILENAME, batch_size, default_csv_data)
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = batched_data

    gender = tf.to_float(tf.equal(sex, ["female"]))
    is_first_class = tf.to_float(tf.equal(pclass, 1))
    is_second_class = tf.to_float(tf.equal(pclass, 2))
    is_third_class = tf.to_float(tf.equal(pclass, 3))
    is_southampton = tf.to_float(tf.equal(embarked, "s"))
    is_queenstown = tf.to_float(tf.equal(embarked, "Q"))
    is_cherbourg = tf.to_float(tf.equal(embarked, "C"))
    features = tf.stack([gender, age, sibsp, parch, is_first_class, is_second_class, \
                         is_third_class, is_southampton, is_queenstown, is_cherbourg])
    #矩阵转秩
    features = tf.transpose(features)
    labels = tf.reshape(survived, [batch_size, 1])
    return features, labels


def _main():
    features, labels = make_inputs(BATCH_SIZE)
    train_model = model.train_model()

    tf.summary.scalar("loss", train_model.loss)
    merged_summary = tf.summary.merge_all()

    graph = tf.get_default_graph()  
    input_graph_def = graph.as_graph_def()  
    output_node_names = ['inputs', 'predict']

    with tf.Session() as sess:
        train_summary = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(10000):
            input_data, input_label = sess.run([features, labels])
            feed_dict = {
                train_model.inputs: input_data,
                train_model.label: input_label,
            }
            _, summary = sess.run([train_model.optimize,merged_summary], feed_dict)
            train_summary.add_summary(summary, i)
            
        coord.request_stop()
        coord.join(threads)
        train_summary.close()

        # save pb file
        output_graph_def = graph_util.convert_variables_to_constants(  
            sess,   
            input_graph_def,   
            output_node_names, # We split on comma for convenience  
        )   

        # Finally we serialize and dump the output graph to the filesystem  
        with tf.gfile.GFile(FROZEN_DIR + FROZEN_FILE, "wb") as f:  
            f.write(output_graph_def.SerializeToString())  
            print("%d ops in the final graph." % len(output_graph_def.node))     


if __name__ == "__main__":
    if os.path.isdir(SUMMARY_DIR):
        os.system("rm -rf %s" % SUMMARY_DIR)

    if os.path.isdir(FROZEN_DIR):
        os.system("rm -rf %s" % FROZEN_DIR)
    os.makedirs(FROZEN_DIR)

    _main()


