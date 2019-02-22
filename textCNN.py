# 使用textCNN对文本进行分类

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec

class TextCNN(nn.Module):
    def __init__(self,dim,words_dict):
        super(TextCNN,self).__init__()
        kernel_size = 4
        self.embedding=nn.Embedding.from_pretrained(words_dict)
        self.conv1d=nn.Conv1d(dim,1,kernel_size)
        self.pool=nn.MaxPool1d(kernel_size)
        self.fc = nn.Linear(dim,dim)
        self.dropout = nn.Dropout()
        self.sm = nn.Softmax()

    def forward(self,sentence):
        sentence=self.conv1d(sentence)
        sentence=self.pool(sentence)
        sentence = self.fc(sentence)
        sentence = self.dropout(sentence)
        return self.sm(sentence)
        # x=torch.cat(x,0)
        # return self.fc(x)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wordsvec = Word2Vec.load('./data/g2.400.model')
    keyvec = wordsvec.wv
    del wordsvec
    wd = torch.FloatTensor(keyvec.vectors)
    model = TextCNN(keyvec.vector_size,wd)
    model.train()
    