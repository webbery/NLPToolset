# 使用textCNN对文本进行分类

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import extract_ann
from sklearn.model_selection import train_test_split
import numpy as np
import time,sys

class TextCNN(nn.Module):
    def __init__(self,vec_dim,sentence_size,label_size):
        super(TextCNN,self).__init__()
        kernel_size = 5
        # out_channel = 32
        self.conv1 = nn.Sequential(
            nn.Conv1d(vec_dim,64,kernel_size,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=1,padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64,32,kernel_size,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=1,padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32,64,kernel_size,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=1,padding=1)
        )
        # self.pool=nn.MaxPool1d(kernel_size)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(sentence_size*64,label_size)
        self.sm = nn.Softmax(0)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        # x=self.pool(x)
        x = x.view(x.size(0),-1)
        # print(x.shape)
        x = self.dropout(x)
        x = self.fc(x)
        # print(x.shape)
        # return self.sm(x)
        return x
        # x=torch.cat(x,0)
        # return self.fc(x)

if __name__ == '__main__':
    # m = nn.Conv1d(16, 33, 3, stride=1)
    # input = torch.randn(20, 16, 50)
    # output = m(input)
    # print(output.shape)

    print(torch.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # td = torch.randn(1,2)
    # td=td.to(device)
    # print(td.size(0))
    # print(torch.sum(td))
    # model_file = sys.argv[1]
    model_file = '../data/f100.zh.model'

    wordsvec = Word2Vec.load(model_file)
    keyvec = wordsvec.wv
    del wordsvec
    word2idx = {word: i for i,word in enumerate(keyvec.index2word)}
    words_dict = torch.FloatTensor(keyvec.vectors)
    embedding=nn.Embedding.from_pretrained(words_dict)
    # print(embedding.embedding_dim)

    batch_size=256
    epoch=0
    
    sentence_size = 1200
    dist_y = extract_ann.generate_target('thuc')
    targets = np.unique(dist_y)

    model = TextCNN(keyvec.vector_size,sentence_size,len(targets))
    model = model.to(device)
    params_to_update = model.parameters()
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()

    plt.ion()
    # plt.ioff()
    fig1 = plt.figure()
    # l, = plt.plot([], [], 'r.')
    # axes = plt.gca()
    x = []
    y = []
    # line, = axes.plot(x, y, 'r-')
    # fig.canvas.draw()
    loss_y=[]
    for (train_x,label) in extract_ann.generate_trainset(batch_size,sentence_size,embedding,word2idx,dist_y,'thuc'):
        train_x = train_x.to(device)
        label = label.to(device)
        # print(len(label))
        epoch+=1
        pred = model(train_x)
        accuracy = 0
        # print(pred.shape)
        # print(pred)
        loss = criterion(pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(target.shape)
        acc = (label == torch.max(pred,1)[1].data.squeeze())
        accuracy = acc.cpu().numpy().sum() / (label.size(0))
        # acc_avg = accuracy/batch_size
        loss_val=loss.cpu().data.double()
        print("epoch: "+str(epoch)+", accuracy: "+ str(accuracy)+", loss: "+str(loss_val))
        x.append(epoch)
        y.append(accuracy)
        loss_y.append(loss_val)
        if epoch%100==0:
            plt.plot(x,y,'-r')
            plt.draw()
            plt.pause(0.0001)
    plt.show()
    model.save('textCNN.model')

    