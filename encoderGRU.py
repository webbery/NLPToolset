import torch
import torch.nn as nn

class EncoderGRU(nn.Module):
    def __init__(self,x_dim,y_dim,layer_num):
        super(EncoderGRU, self).__init__()
        self.gru = nn.GRU(input_size=x_dim,hidden_size=y_dim,num_layers=layer_num)

    def forward(self, x,h=None):
        # packed = nn.utils.rnn.pack_padded_sequence(x,x_len)
        o,h = self.gru(x)
        # o,_ = nn.utils.rnn.pad_packed_sequence(o)
        return o,h