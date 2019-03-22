import torch
import torch.nn as nn

class Encoder():
    def __init__(self,x_dim,y_dim,layer_num):
        self.gru = nn.GRU(input_size=x_dim,output_size=y_dim,num_layers=layer_num)

    def result(self,input):
        return self.gru(input)