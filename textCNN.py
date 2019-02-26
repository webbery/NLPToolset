# 使用textCNN对文本进行分类

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec
import extract_ann

class TextCNN(nn.Module):
    def __init__(self,dim):
        super(TextCNN,self).__init__()
        kernel_size = 4
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
    print(torch.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wordsvec = Word2Vec.load('./data/g2.400.model')
    keyvec = wordsvec.wv
    del wordsvec
    word2idx = {word: i for i,word in enumerate(keyvec.index2word)}
    words_dict = torch.FloatTensor(keyvec.vectors)
    embedding=nn.Embedding.from_pretrained(words_dict)

    batch_size=1280
    epoch=1000
    
    model = TextCNN(keyvec.vector_size)
    model = model.to(device)
    params_to_update = model.parameters()
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for (train_x,target) in extract_ann.generate_trainset(batch_size,20,embedding,word2idx):
        # train_x = train_x.to(device)
        pred = model(torch.FloatTensor(train_x))
        # target.to(device)
        loss = criterion(pred,target)
        optimizer.zero_grad()
        optimizer.step()

    