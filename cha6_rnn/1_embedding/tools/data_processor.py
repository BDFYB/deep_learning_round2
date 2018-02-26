import bz2
import random
import numpy as np

PAGE_DATA_DIR = "../data/pages.bz2"
VOCAL_DATA_DIR = "../data/vocabulary.bz2"

# 制造pages迭代器
def data_iterator():

    # vocabulary: 制造的字典，word:num这种形式
    with bz2.open(VOCAL_DATA_DIR, 'rt') as vocabulary:
        print('Read vocabulary')
        vocabulary = [x.strip() for x in vocabulary]
        #print(vocabulary)
    indicy = {x: i for i, x in enumerate(vocabulary)}

    # 每次循环返回一个page（该page为从Wikipedia获取的处理后的网页数据）
    with bz2.open(PAGE_DATA_DIR, 'rt') as pages:
        for page in pages:
            words = page.strip().split()
            words = [indicy.get(word, 0) for word in words]
            #word转为vocabulary的one hot位置数字
            yield words

#制造skip grams迭代器, 由当前词推断上下文。max_context: 最大上下文个数
def make_skipgram_data(max_context):
    for page_word_list_in_num in data_iterator():
        """page_word_list_in_num:一个用index表示的list, eg:[2602, 5302, 0, 3,......]
        """
        """
        对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），
        enumerate将其组成一个索引序列，利用它可以同时获得索引和值
        enumerate多用于在for循环中得到计数
        """
        for index, current_word in enumerate(page_word_list_in_num):
            context = random.randint(1, max_context)
            for target in page_word_list_in_num[max(0, index - context): index]:
                yield current_word, target
            for target in page_word_list_in_num[index + 1: index + 1 + context]:
                yield current_word, target

def get_batched_data(max_context, batch_size):
    input_data = np.zeros(batch_size)
    target = np.zeros(batch_size)
    iterator = make_skipgram_data(max_context)
    while True:
        for index in range(batch_size):
            input_data[index], target[index] = next(iterator)
        yield input_data, target


if __name__ == "__main__":
    for i in data_iterator():
        #print(i)
        count = 0
        for cur, target in get_batched_data(2, 4):
            count += 1
            print(cur, target)
            if count > 5:
                break
        break