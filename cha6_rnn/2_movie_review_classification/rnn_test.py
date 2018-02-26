import tensorflow as tf
import numpy as np


STATIC_TEST_SUMMARY_DIR = "./static_test_summary"
DYNAMIC_TEST_SUMMARY_DIR = "./dynamic_test_summary"

def build_model(num_units, batch_size):
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units)
    state = cell.zero_state(batch_size, tf.float32)
    return cell, state

def main():
    num_units = 8
    batch_size = 2
    # 4 * 7 4个单词，字典数为7个，onehot标识
    input_data = [[[1, 0, 0, 0, 0, 0, 0], 
                   [0, 1, 0, 0, 0, 0, 0]], 
                  [[0, 0, 0, 1, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 1, 0]],
                  [[0, 1, 0, 0, 0, 0, 0], 
                   [0, 0, 1, 0, 0, 0, 0]], 
                  [[0, 0, 0, 1, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 1, 0]]]

    # 3 * 7 3个单词 和一个2 * 7，两个单词，第三个padding为0了的input
    input_data2 = [[[1, 0, 0, 0, 0, 0, 0], 
                   [0, 1, 0, 0, 0, 0, 0]], 
                  [[0, 0, 0, 1, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 1, 0]],
                  [[0, 1, 0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0]]]

    # 模拟两个循环
    total_inputs = [input_data, input_data2]

    cell, state = build_model(num_units, batch_size)
    """
    # 手动实现rnn动力学
    print("start handwrite rnn")
    input_place = tf.placeholder(shape=(None, 7), dtype=tf.float32)
    with tf.Session() as sess:
        output, state = cell(input_place, state)
        sess.run(tf.global_variables_initializer())
        # 手动实现rnn动力学
        for i in range(4):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            print("step %s" % i)
            feed_dict = {
                # 一次只能塞一行
                input_place: [input_data[i][0], input_data[i][1]],
            }
            outs, outs_s = sess.run([output, state], feed_dict=feed_dict)
            print("output: \n%s" % outs)
            # state 与 output一样
            print("out state: \n%s" % outs_s)
    """
    """
    # static_rnn
    # 注意：这里面识别input为batch_size = 2，embedding_size = 7，sequence_length = 4
    # static_rnn对应的inputs为 sequence_length * batch_size * embedding_size
    # 输入需要为tensor，不能是list。可以用constant转。另外输入必须是个sequence，即列表
    print("start static_rnn")
    batch_size = 2
    embedding_size = 7
    max_sequence_length = 4

    inputs = []
    for x in range(max_sequence_length):
        place_name = "input_%s" % x
        inputs.append(tf.placeholder(shape=(None, embedding_size), name=place_name, dtype=tf.float32))
    sequence_length_holder = tf.placeholder(shape=(2), dtype=tf.int32)

    #output, state = tf.nn.static_rnn(cell, inputs, dtype=tf.float32)
    output, state = tf.nn.static_rnn(cell, inputs, sequence_length=sequence_length_holder, initial_state=state, dtype=tf.float32)
    with tf.Session() as sess:    
        sess.run(tf.global_variables_initializer())
        static_sum = tf.summary.FileWriter(STATIC_TEST_SUMMARY_DIR, sess.graph)
        for count, input_data_single in enumerate(total_inputs):
            if count == 0:
                feed_dict = {
                    sequence_length_holder: [4, 4],
                }
            else:
                feed_dict = {
                    sequence_length_holder: [3, 2],
                }                
                input_data_single.append([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])

            for count, input_data in enumerate(input_data_single):
                name="input_%s:0" % count
                feed_dict[name] = input_data
            print("start one loop")
            outs, outs_s = sess.run([output, state], feed_dict=feed_dict)
            print("output: \n%s" % outs)
            # state 与 output一样
            print("out state: \n%s" % outs_s)
    """
    # dynamic_rnn
    # time_major = true, 则格式与static_rnn一致，sequence_length * batch_size * embedding_size
    # time_major 默认为false，符合一般情况，batch_size * sequence_length * embedding_size 
    # batch_size是构建计算图就需要确定好的
    # 本例数据说明了相邻两次input长度不一样的情况以及同一个input多batch每个length都不同的情况
    print("start dynamic_rnn")
    dynamic_input_place_holder = tf.placeholder(shape=(None, 2, 7), name="dynamic_input", dtype=tf.float32)
    seq_length = tf.placeholder(shape=(2), name="sequence_length", dtype=tf.int32)
    output, state = tf.nn.dynamic_rnn(cell, dynamic_input_place_holder, sequence_length=seq_length, time_major=True, dtype=tf.float32)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dynamic_sum = tf.summary.FileWriter(DYNAMIC_TEST_SUMMARY_DIR, sess.graph)

        count = 0
        for input_data_single in total_inputs:
            print("start one loop")
            count += 1
            if count == 1:
                length = [4, 4]
            else:
                length = [3, 2]

            feed_dict = {
                dynamic_input_place_holder: input_data_single,
                seq_length: length,
            }
            outs, outs_s = sess.run([output, state], feed_dict=feed_dict)
            print("output: \n%s" % outs)
            # state 为最后一个state
            print("out state: \n%s" % outs_s)        
    

if __name__ == '__main__':
    main()