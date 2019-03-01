# 使用textCNN对文本进行分类

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec
import extract_ann

class TextCNN(nn.Module):
    def __init__(self,vec_dim,sentence_size,label_size):
        super(TextCNN,self).__init__()
        kernel_size = 4
        # out_channel = 50
        self.conv1d=nn.Conv1d(vec_dim,1,kernel_size)
        self.pool=nn.MaxPool1d(kernel_size)
        self.fc = nn.Linear(kernel_size,label_size)
        # self.dropout = nn.Dropout()
        self.sm = nn.Softmax(0)

    def forward(self,x):
        x=self.conv1d(x)
        x=self.pool(x)
        x = self.fc(x)
        # x = self.dropout(x)
        return self.sm(x)
        # x=torch.cat(x,0)
        # return self.fc(x)

if __name__ == '__main__':
    # m = nn.Conv1d(16, 33, 3, stride=1)
    # input = torch.randn(20, 16, 50)
    # output = m(input)
    # print(output.shape)
    print(torch.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    for (train_x,label) in extract_ann.generate_trainset(batch_size,sentence_size,embedding,word2idx,targets):
        train_x = train_x.to(device)
        label = label.to(device)
        # print(train_x.shape)
        print("epoch: "+str(epoch))
        epoch+=1
        for i in range(0,batch_size):
            pred = model(train_x)
            # print(pred.shape)
            print(pred[i].shape)
            print(pred[i][0].shape)
            print(pred[i][0].dim())
            # print(label.shape)
            # print(label[i].shape)
            # print(label[i][0].shape)
            # print('---------')
            # print(label[i][0])
            loss = criterion(pred[i][0],label[i][0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    