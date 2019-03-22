# 实现论文Sequence to Sequence Learningwith Neural Networks
import torch
import torch.nn as nn
import torch.optim as optim
import encoder

class Seq2Seq(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder.Encoder(input_dim,output_dim,4)

    def forward(self, x,h=None):
        x,h= self.encoder(x,h)
        return x,h

#句子按词翻转逆序