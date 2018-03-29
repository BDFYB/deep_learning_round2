import tensorflow as tf 
import numpy as np
import json
import os

DATA_REAL_PATH = "../data/arxiv.txt"

class DataPool(object):

    VOCABULARY = \
        " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
        "\\^_abcdefghijklmnopqrstuvwxyz~<>{|}"

    def __init__(self, data_file, batch_size, sentence_length):
        if not os.path.isfile(data_file):
            raise Exception("data file %s not exsists!" % data_file)
        with open(data_file) as fd:
            self.data = fd.readlines()
        self.batch_size = batch_size
        self.vocab_len = len(DataPool.VOCABULARY)
        self.sentence_length = sentence_length
        self.lookup_dict = {x: i for i, x in enumerate(DataPool.VOCABULARY)}
        self.back_lookup_dict = {i: x for i, x in enumerate(DataPool.VOCABULARY)}

    def __iter__(self):
        #此方法用来测试
        for text in self.data:
            str_list = text.replace("\n", "").split(".")
            for strs in str_list:
                if strs == "":
                    continue
                yield strs + '. '

    def __next__(self):
        #按照句子来喂，一个句子是一条数据
        count = 0
        ret_data_one_hot = np.zeros([self.sentence_length, self.batch_size, self.vocab_len])
        #一行语句，包括多个句子
        for sentence in self.data:
            str_list = sentence.replace("\n", "").split(".")
            for strs in str_list:
                #print(len(strs))
                #对于每一个句子，作为一条数据
                if strs == "":
                    continue
                #因为在前面按照.进行句子分割，在这里面每句话加上了.和空格，因此实际长度为句子长度+2
                strs += '. '
                char_pos = 0
                for char in strs:
                    try:
                        char_index =  self.lookup_dict[char]
                    except Exception as e:
                        #print("error: %s, sentence:%s, char:%s" % (e, strs, char))
                        print("error: %s, char:%s" % (e, char))
                        continue

                    ret_data_one_hot[char_pos][count][char_index] = 1
                    char_index += 1
                    char_pos += 1   
                count += 1
                if count == self.batch_size:
                    yield ret_data_one_hot
                    ret_data_one_hot = np.zeros([self.sentence_length, self.batch_size, self.vocab_len])
                    count = 0             



if __name__ == "__main__":
    batch_size = 2
    sentence_length = 1140
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_REAL_PATH)
    data_pool = DataPool(file_path, batch_size, sentence_length)

    """
    #获取序列最大长度
    pool_iter = iter(data_pool)
    count = 0
    max_length = 0
    for text in pool_iter:
        count += 1
        #print(text)
        current_length = len(text)
        if current_length > max_length:
            max_length = current_length
    #测试最大长度是1138
    print("sentence max length: %s" % max_length)
    
    """
    # 数据获取
    for index, text_in_onehot in enumerate(next(data_pool)):
        
        #print(text_in_onehot)
        #print("new round")
        seq1 = ""
        cur_batch = 0
        # 从数字转为string例子
        for seq in text_in_onehot:
            # 打印每个batch的第二个数据
            lookups = data_pool.back_lookup_dict[np.argmax(seq.tolist()[1])]
            seq1 += lookups
            if lookups == '.':
                seq1 += " "
                break
        print(seq1)

        #break




