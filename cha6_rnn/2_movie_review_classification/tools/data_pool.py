import os
import bz2
import random
import numpy

"""
功能描述：
1、指定字典（embedding的为10000大小，自带字典为80000+大小）
2、指定batch_size（未实现）
3、输出：返回一个可迭代对象，为已根据字典解析成对应index的list和label，label为0的是负面评价
为1的是正面评价，并且通过注释可以指定是输出定长数据还是输出不定长数据（即全部词汇构成的list）
"""
DICTIONARY_ORI = "/../data/aclImdb/imdb.vocab"
DICTIONARY_EMBEDDING = "/Users/baidu/AI/deep_learning_round2/cha6_rnn/1_embedding/data/vocabulary.bz2"
SENTENCE_MAX_LENGTH = 2000


class DataPool(object):
    def __init__(self, data_top_dir, is_use_embedding=False, batch_size=1):
        self.batch_size = batch_size
        neg_dir = data_top_dir + '/neg'
        pos_dir = data_top_dir + '/pos'
        if not os.path.isdir(neg_dir) or\
           not os.path.isdir(pos_dir):
           raise Exception("subdir not exsists!")

        # 1、制作filelist
        self.file_list = []
        for file in os.listdir(neg_dir):
            file_path = os.path.join(neg_dir, file)
            if not os.path.isdir(file_path):
                self.file_list.append({file_path: 0})

        for file in os.listdir(pos_dir):
            file_path = os.path.join(pos_dir, file)
            if not os.path.isdir(file_path):
                self.file_list.append({file_path: 1})              

        random.shuffle(self.file_list)

        # 2 读取字典
        if is_use_embedding:
            data_vocab_dir = DICTIONARY_EMBEDDING
            with bz2.open(data_vocab_dir, 'rt') as vocab:
                print('Read vocabulary embedding')
                vocabulary = [x.strip() for x in vocab]
                print('dictionary length: %s' % len(vocabulary))
        else:
            data_vocab_dir = os.path.split(os.path.realpath(__file__))[0] + DICTIONARY_ORI
            with open(DICTIONARY_ORI) as vocab:
                print('Read vocabulary ori')
                vocabulary = [x.strip() for x in vocab]
                print('dictionary length: %s' % len(vocabulary))
        self.vocab_map = {}
        """
        self.vocab_map["UNDEF"] = 0
        self.vocab_map["ST_TAG"] = 1
        self.vocab_map["END_TAG"] = 2
        """
        for index, key in enumerate(vocabulary):
            self.vocab_map[key] = index + 1


    def __iter__(self):
        return self


    def __next__(self):
        for data_dict in self.file_list:
            vocab_list_in_num = numpy.zeros(SENTENCE_MAX_LENGTH)
            for file_path, label in data_dict.items():
                with open(file_path, 'r') as file_fd:
                    #print(file_path)
                    content_str = file_fd.read().replace('.', '').replace('"', "")
                    content_str = content_str.replace(',', '').replace("'", "")
                    content_str = content_str.replace('<br />', "")

                    vocab_list = content_str.split(' ')                    
                    # 粗暴点。这里做了一般的简单字符串替换，然后字典中不存在的就都取为0。这里是非固定长度数据
                    # yield [self.vocab_map.get(x.lower(), 0) for x in vocab_list], label

                    prepare_list = [self.vocab_map.get(x.lower(), 0) for x in vocab_list]
                    #print(prepare_list)
                    for index, data in enumerate(prepare_list):
                        if index == SENTENCE_MAX_LENGTH:
                            break
                        vocab_list_in_num[index] = data
                    yield vocab_list_in_num, label

                    # 用 map函数也可以达到同样目的
                    #yield map([lambda x: self.vocab_map.get(x.lower(), 0)], vocab_list), label 
                break          


if __name__ == "__main__":
    DATA_DIR = '../aclImdb/test'
    BATCH_SIZE = 1
    data_pool = DataPool(DATA_DIR, True)
    max_length = 0
    for sentence, label in next(data_pool):
        print(sentence)
        
        length = len(sentence)
        if length > max_length:
            max_length = length
        break
    #数据最大一句话长度 2192
    print("max sentence length: %s" % max_length)

        