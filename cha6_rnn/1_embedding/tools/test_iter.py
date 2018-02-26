import time


class DataPool(object):

    def __init__(self, count):
        self.list = [x for x in range(count)]

    #表明该类是一个可迭代对象
    def __iter__(self):
        return self

    #next方法将调用这个
    def __next__(self):
        for i in self.list:
            yield i

if __name__ == "__main__":
    count = 10
    pool = DataPool(count)
    #for i in next(pool):
    for i in pool:
        #for x in i:
        #    print(x)
        time.sleep(1)
        print(i)