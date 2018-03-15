#!/usr/bin/python
import os
import csv
import random
import numpy as np

"""
可迭代对象，每次迭代输出为：
data: word中一系列字母图像展平后的序列，尺寸：
    MAX_WORD_LENGTH * BATCH_SIZE * SINGLE_IMAGE_LENGTH
label: 转化为数字的字母，尺寸：
    MAX_WORD_LENGTH * BATCH_SIZE
"""

# 每个字母的格式：16 * 8 = 128位
# 每个Word的最大长度为14 
DATA_REAL_PATH = "../data/letter.data"
BATCH_SIZE = 2
MAX_WORD_LENGTH = 14
SINGLE_IMAGE_LENGTH = 16 * 8

class DataPool(object):
    def __init__(self, data_abs_path, batch_size, max_word_length, image_length):
        if not os.path.isfile(data_abs_path):
            raise Exception("file %s not exsists!" % data_abs_path)
        self.data_path = data_abs_path
        self.batch_size = batch_size
        self.total_data_in_lines = self._data_pre_loading()
        self.letter_map = LetterMap()
        self.max_word_length = max_word_length
        self.single_image_length = image_length

    def _data_pre_loading(self):
        with open(self.data_path) as fd:
            lines = list(csv.reader(fd, delimiter='\t'))
            # 为保证训练不被数据影响，这里random_shuffle一下
            # pass 不能在这里shuffle，因为这里的数据只是一个字母，应该按单词shuffle
            #random.shuffle(lines)
        return lines

    def _parse_lines(self, line):
        # 由于格式原因最后一个是空格
        data = line[6:-1]
        data = [int(x) for x in data]
        return line[1], int(line[2]), int(line[3]), int(line[4]), data

    def __iter__(self):
        return self

    def __next__(self):
        ret_label = np.zeros(shape=(self.max_word_length, self.batch_size, self.letter_map.label_size))
        ret_data = np.zeros(shape=(self.max_word_length, self.batch_size, self.single_image_length))
        current_batch = 0
        strx = ""
        for line in self.total_data_in_lines:
            letter, next_id, word_id, position, got_data = self._parse_lines(line)
            strx += letter
            letter_num = self.letter_map.get_num_by_letter(letter)

            ret_label[position-1][current_batch][letter_num] = 1
            for index, single_data in enumerate(got_data):
                ret_data[position-1][current_batch][index] = single_data
            if next_id == -1:
                #print(strx)
                strx = ""
                current_batch += 1
                if current_batch == self.batch_size:
                    yield ret_data, ret_label
                    current_batch = 0
                    ret_label = np.zeros(shape=(self.max_word_length, self.batch_size, self.letter_map.label_size))
                    ret_data = np.zeros(shape=(self.max_word_length, self.batch_size, self.single_image_length))

    def get_max_word_length(self):
        """
        获取最大Word长度
        """
        max_word_length = 0
        for line in self.total_data_in_lines:
            letter, next_id, word_id, position, data = self._parse_lines(line)
            if position > max_word_length:
                max_word_length = position
        print(max_word_length)

class LetterMap(object):
    """
    小写a-z输出对应数字(train用作label)及根据对应数字获得字母(infer展示)
    a 对应为0
    """
    def __init__(self):
        self.base_letter = 'a'
        self.base_letter_num = ord(self.base_letter)
        self.max_letter_num = ord('z')
        self.num_units = self.max_letter_num - self.base_letter_num + 1

    @property
    def label_size(self):
        return self.num_units

    def get_num_by_letter(self, letter):
        return int(ord(letter) - self.base_letter_num)

    def get_letter_by_num(self, num):
        num = self.base_letter_num + int(num)
        if num < self.base_letter_num or num > self.max_letter_num:
            ret = ""
        else:
            ret = chr(num)
        return ret

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_REAL_PATH)
    data_pool = DataPool(file_path, BATCH_SIZE, MAX_WORD_LENGTH, SINGLE_IMAGE_LENGTH)
    #data_pool.get_max_word_length()
    # 测试letter_map
    letter_map = LetterMap()
    #print(letter_map.get_letter_by_num(letter_map.get_num_by_letter('z')))

    for batched_data, batched_label in next(data_pool):
        print(batched_data)
        print(batched_label)
        break
        
    #验证
    strx = ""
    for single_data in batched_label:
        #取第一个元素（第一个batch的label）
        single_data = single_data[0].tolist()
        if 1 not in single_data:
            continue
        strx += letter_map.get_letter_by_num(single_data.index(max(single_data)))
    print(strx)


