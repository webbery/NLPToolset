import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import codecs
import os

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN,self).__init__()
        kernel_size = 2
        self.embedding=nn.Embedding(10000,300)
        self.conv1d=nn.Conv1d(1,1,kernel_size)
        self.pool=nn.MaxPool1d(kernel_size)
        self.fc=nn.Linear(300,300)

    def forward(self,x):
        x=self.conv1d(x)
        x=self.pool(x)
        x=torch.cat(x,0)
        return self.fc(x)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
                
# tcnn=TextCNN()
# weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
# embedding = nn.Embedding.from_pretrained(weight)
# input = torch.LongTensor([1])
# val = embedding(input)
# print(val)

# wf = codecs.open('./data/wiki.zh.word.2.text','w','utf-8')
# f = codecs.open('./data/wiki.zh.word.text', 'r','utf-8')
# line = f.readline()
# lastline=""
# while line:
#     try:
#         if lastline!=line:
#             wf.write(line)
#             lastline = line
#         line = f.readline()
#     except UnicodeDecodeError:
#         print('error')
#         f.seek(2,1)
# f.close()
# wf.close()

sentences = LineSentence('./data/wiki.zh.word.2.text')
# print(sentences)
model  = Word2Vec(sentences,size = 800,hs = 1,window =3)
model.save('./data/wiki.zh.800.model')
print(model.most_similar('科学家'))
