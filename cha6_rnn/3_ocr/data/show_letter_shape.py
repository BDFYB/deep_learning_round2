#!/bin/python 

data = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 0 0 1 0 0 1 0 1 1 0 1 1 1 0 0 0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0".split(" ")

tmp_data = []
for count, i in enumerate(data):
    if count % 8 == 0:
        tmp_str = ""
        for single in tmp_data:
            tmp_str += single
            tmp_str += " "
        print(tmp_str)
        #print(tmp_data)
        tmp_data = []
    tmp_data.append(i)

