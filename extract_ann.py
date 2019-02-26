# 提取数据集中标注信息y,x 并以词向量的方式表示句子

import json
import os.path
import sys
import seg_sentence
import torch
import torch.nn as nn
import numpy

# return y,x
def get_trainset_data():
    primitive_corpus = open('./data/web_text_zh_train.json', 'r',encoding='utf8')
    line = primitive_corpus.readline()
    while line:
        jstr = json.loads(line)
        yield jstr['topic'],jstr['content']
        line = primitive_corpus.readline()
    primitive_corpus.close()

# def add_null_vector(arr,size):
#     vec = numpy.zeros(size)
#     arr.append(vec)
#     return arr

# return train_x,train_y
def generate_trainset(batch_size,sentence_len,embedding,word2idx):
    train_x=[]
    train_y=[]
    print(len(word2idx))
    for (y,x) in get_trainset_data():
        array_y = seg_sentence.segment(y,type="arr")
        array_x = seg_sentence.segment(x,type="arr")
        for idx in range(0,sentence_len):
            if idx >=len(array_y):
                vec_y = numpy.zeros(400)
                train_y.append(vec_y)
            else:
                word = array_y[idx]
                if word in word2idx:
                    vec_y = embedding.weight[word2idx[word]]
                    train_y.append(vec_y)
                else:
                    vec_y = numpy.zeros(400)
                    train_y.append(vec_y)
            if idx >=len(array_x):
                vec_x = numpy.zeros(400)
                train_x.append(vec_x)
            else:
                word = array_x[idx]
                if word in word2idx:
                    vec_x = embedding.weight[word2idx[word]]
                    train_x.append(vec_x)
                else:
                    vec_x = numpy.zeros(400)
                    train_x.append(vec_x)
        batch_size-=1
        if batch_size<0:
            yield train_x,train_y
            train_x.clear()
            train_y.clear()
