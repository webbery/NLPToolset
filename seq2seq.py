# 实现论文Sequence to Sequence Learningwith Neural Networks
# 数据集使用WMT'14 English-German data (https://nlp.stanford.edu/projects/nmt/)
import torch
import torch.nn as nn
import torch.optim as optim
import encoderGRU
import decoderGRU
from CustomEmbedding import CustomEmbedding
from sentences import Sentences
import os, sys

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x,h=None):
        x,h= self.encoder(x,h)
        return x,h

    def set_mode(self,mode='train'):
        if mode=='train':
            self.encoder.train()
            self.decoder.train()

#数据预处理  
#初始化词向量
model_file = sys.argv[1]
embedding = CustomEmbedding(model_file)
flag_words_end=embedding.get_length()
print(flag_words_end)
speical_words=['GO','EOS','UNK']
speical_words_index = [flag_words_end+1,flag_words_end+2,flag_words_end+3]
#读取英文输入
input_data = Sentences('./data/wmt14/train.en')
# input_data.readline(reversed=True)
# input_file = open('./data/wmt14/train.en', 'r',encoding='utf8')
# line = input_file.readline()
# sentences = []
# while line:
#     #向量化
#     words = seg_sentence.segment(line)
#     vectors = []
#     for word in words:
#         input = embedding.get_index(word)
#         vectors.append(input)
#     #句子按词翻转逆序
#     vectors.reverse()
#     vectors.append(speical_words_index[0])  # add 'GO'
#     sentences.append(vectors)
#     line = input_file.readline()
# print('load english file finished')
#读取目标
target_data = Sentences('./data/wmt14/train.de')
# target_file = open('./data/wmt14/train.de', 'r',encoding='utf8')
# line = target_file.readline()
# targets = []
# while line:
#     #向量化
#     words = seg_sentence.segment(line)
#     vectors = []
#     for word in words:
#         #向量化
#         input = embedding.get_index(word)
#         vectors.append(input)
#     vectors.append(speical_words_index[1])  # add 'EOS'
#     targets.append(vectors)
#     line = target_file.readline()
# print('load translate file finished')
#
input_dim = 100
output_dim = 500
encoderGRU = encoderGRU.EncoderGRU(input_dim,output_dim,4)
decoderGRU = decoderGRU.DecoderGRU(output_dim,5)
seq2seq = Seq2Seq(encoderGRU,decoderGRU)
#Train
encoderGRU.train()
decoderGRU.train()
seq2seq.train()

learning_rate = 0.0001
encoder_optimizer = optim.Adam(encoderGRU.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoderGRU.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()
target_gen = target_data.readline()
input_gen = input_data.readline(reverse=True)
line = target_gen.__next__()
while line:
    words = input_gen.__next__()
    input = []
    for word in words:
        idx = embedding.get_index(word)
        if idx==None:
            idx = flag_words_end+2
        input.append(idx)

    label=[]
    for word in line:
        idx = embedding.get_index(word)
        if idx==None:
            idx = flag_words_end+2
        label.append(idx)
    line = target_gen.__next__()

    fixed_vector = encoderGRU(input)
    pred = decoder(fixed_vector)
    loss = criterion(pred,label)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()
    decoder_optimizer.step()
    encoder_optimizer.step()
    print('epoch: '+idx)