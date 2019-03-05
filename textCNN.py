# 使用textCNN对文本进行分类

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import extract_ann
# import numpy as np
import time

class TextCNN(nn.Module):
    def __init__(self,vec_dim,sentence_size,label_size):
        super(TextCNN,self).__init__()
        kernel_size = 4
        # out_channel = 32
        self.conv1 = nn.Sequential(
            nn.Conv1d(vec_dim,64,5),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64,32,kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # self.pool=nn.MaxPool1d(kernel_size)
        self.fc = nn.Linear(2,label_size)
        # self.dropout = nn.Dropout()
        self.sm = nn.Softmax(0)

    def forward(self,x):
        # print(x.shape)
        x=self.conv1(x)
        # print(x.shape)
        x=self.conv2(x)
        # print(x.shape)
        # x=self.pool(x)
        # x = self.dropout(x)
        x = x.view(x.size(),-1)
        # print(x.shape)
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

    wordsvec = Word2Vec.load('./data/g2.400.model')
    keyvec = wordsvec.wv
    del wordsvec
    word2idx = {word: i for i,word in enumerate(keyvec.index2word)}
    words_dict = torch.FloatTensor(keyvec.vectors)
    embedding=nn.Embedding.from_pretrained(words_dict)

    batch_size=128
    epoch=0
    
    sentence_size = 20
    targets = extract_ann.generate_target()

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
    for (train_x,label) in extract_ann.generate_trainset(batch_size,sentence_size,embedding,word2idx,targets):
        train_x = train_x.to(device)
        label = label.to(device)
        epoch+=1
        pred = model(train_x)
        accuracy = 0
        for i in range(0,batch_size):
            pred_i = torch.t(pred[i])
            target = label[i][0]
            loss = criterion(pred_i,target)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            # print(target.shape)
            acc = (target == torch.max(pred_i,1)[1].data.squeeze())
            accuracy += acc.cpu().numpy().sum() / (target.size(0))
        acc_avg = accuracy/batch_size
        print("epoch: "+str(epoch)+", accuracy: "+ str(acc_avg))
        x.append(epoch)
        y.append(acc_avg)
        if epoch%100==0:
            plt.plot(x,y,'-r')
            plt.draw()
            plt.pause(0.0001)
    plt.show()


    