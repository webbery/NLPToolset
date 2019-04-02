import torch
import torch.nn as nn

class DecoderGRU(nn.Module):
    def __init__(self,hidden_size,layer_num):
        super(DecoderGRU, self).__init__()
        self.gru=nn.GRU(hidden_size,hidden_size,layer_num)

    def forward(self, x,x_len,h=None):
        rnn_output, hidden = self.gru(x, h)
        return rnn_output,hidden