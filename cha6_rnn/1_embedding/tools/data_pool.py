import bz2
import random
import numpy as np

PAGE_DATA_DIR = "../data/pages.bz2"
VOCAL_DATA_DIR = "../data/vocabulary.bz2"


class DataPool(object):

    def __init__(self, batch_size, max_context, vocab_dir, page_data_dir):
        self.batch_size = batch_size
        self.max_context = max_context
        self.data_vocab_dir = vocab_dir
        self.data_page_dir = page_data_dir
        self.input_data = np.zeros(batch_size)
        self.target = np.zeros(batch_size)

        # vocabulary: 制造的字典，word:num这种形式
        with bz2.open(self.data_vocab_dir, 'rt') as vocabulary:
            print('Read vocabulary')
            vocabulary = [x.strip() for x in vocabulary]
            #print(vocabulary)
        self.indicy = {x: i for i, x in enumerate(vocabulary)}
        self.vocab_size = len(vocabulary)
        self.iterator = self.make_skipgram_data()

    def __iter__(self):
        return self

    def __next__(self):
        print("start data iterate")

        while True:
            for index in range(self.batch_size):
                self.input_data[index], self.target[index] = next(self.iterator)
            yield self.input_data, self.target

    # 制造pages迭代器
    def data_iterator(self):
        print('Read page')
        # 每次循环返回一个page（该page为从Wikipedia获取的处理后的网页数据）
        with bz2.open(self.data_page_dir, 'rt') as pages:
            for page in pages:
                words = page.strip().split()
                words = [self.indicy.get(word, 0) for word in words]
                #word转为vocabulary的one hot位置数字
                yield words

    #制造skip grams迭代器, 由当前词推断上下文。max_context: 最大上下文个数
    def make_skipgram_data(self):
        print('generate skipgram data')
        for page_word_list_in_num in self.data_iterator():
            """page_word_list_in_num:一个用index表示的list, eg:[2602, 5302, 0, 3,......]
            """
            """
            对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），
            enumerate将其组成一个索引序列，利用它可以同时获得索引和值
            enumerate多用于在for循环中得到计数
            """
            for index, current_word in enumerate(page_word_list_in_num):
                context = random.randint(1, self.max_context)
                for target in page_word_list_in_num[max(0, index - context): index]:
                    yield current_word, target
                for target in page_word_list_in_num[index + 1: index + 1 + context]:
                    yield current_word, target



if __name__ == "__main__":

    pool = DataPool(batch_size=4, max_context=3, vocab_dir=VOCAL_DATA_DIR, page_data_dir=PAGE_DATA_DIR)
    count = 0
    for i in next(pool):
        print("timex")
        print(i)
        count += 1
        if count == 3:
            break
