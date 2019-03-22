# 提取数据集中标注信息y,x 并以词向量的方式表示句子

import json
import os.path
import sys
import seg_sentence
import torch
import torch.nn as nn
import numpy
import matplotlib.pyplot as plt

def generate_target_file():
    primitive_corpus = open('./data/web_text_zh_train.json', 'r',encoding='utf8')
    labels = open('./data/labels.txt','w',encoding='utf8')
    line = primitive_corpus.readline()
    target={}
    label=0
    while line:
        jstr = json.loads(line)
        k = jstr['topic']
        # keys = seg_sentence.segment(key,type="arr")
        # k = keys[0]
        if k not in target:
            target[k]=label
            labels.write(str(label)+" "+k+"\n")
            label += 1
            # print(k+","+str(label))
        line = primitive_corpus.readline()
    primitive_corpus.close()
    labels.close()
    return target

def generate_target():
    primitive_corpus = open('./data/web_text_zh_train.json', 'r',encoding='utf8')
    line = primitive_corpus.readline()
    target={}
    label=0
    while line:
        jstr = json.loads(line)
        k = jstr['topic']
        if k not in target:
            target[k]=label
            label += 1
        line = primitive_corpus.readline()
    primitive_corpus.close()
    return target

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

def show_data_destribute():
    # target_len = len(targets)
    sentence_length=[]
    l_x=[]
    idx = 0
    for (y,x) in get_trainset_data():
        array_x = seg_sentence.segment(x,type="arr")
        sentence_length.append(len(array_x))
        idx+=1
        l_x.append(idx)
        if idx>10000:
            break
    # plt.hist(sentence_length)
    plt.plot(l_x,sentence_length,'+')
    plt.show()
    plt.savefig("dist.png")

# return train_x,train_y
def generate_trainset(batch_size,sentence_len,embedding,word2idx,targets):
    train_x=torch.zeros([batch_size,400,sentence_len],dtype=torch.float32)
    target_len = len(targets)
    print('target_len:' + str(target_len))
    train_y=torch.zeros([batch_size,1,target_len],dtype=torch.long)
    group_idx=0
    for (y,x) in get_trainset_data():
        array_x = seg_sentence.segment(x,type="arr")
        tensor_x = torch.zeros([sentence_len,400],dtype=torch.float32)
        # generate tensor X
        for idx in range(0,sentence_len):
            if idx <len(array_x):
                word = array_x[idx]
                if word in word2idx:
                    vec_x = embedding.weight[word2idx[word]]
                    tensor_x[idx]=vec_x
        # generate label
        tensor_y = torch.zeros([target_len,1],dtype=torch.long)
        tensor_y[targets[y]]=1

        train_x[group_idx]=torch.t(tensor_x)
        train_y[group_idx]=torch.t(tensor_y)
        group_idx+=1
        if group_idx>=batch_size:
            yield train_x,train_y
            train_x=torch.zeros([batch_size,400,sentence_len],dtype=torch.float32)
            train_y=torch.zeros([batch_size,1,target_len],dtype=torch.long)
            group_idx = 0

def get_validset_data():
    primitive_corpus = open('./data/web_text_zh_valid.json', 'r',encoding='utf8')
    line = primitive_corpus.readline()
    while line:
        jstr = json.loads(line)
        yield jstr['topic'],jstr['content']
        line = primitive_corpus.readline()
    primitive_corpus.close()