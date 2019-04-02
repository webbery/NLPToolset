import torch.nn as nn
from gensim.models import Word2Vec
import torch

class CustomEmbedding():
    def __init__(self,model_file):
        wordsvec = Word2Vec.load(model_file)
        keyvec = wordsvec.wv
        del wordsvec
        self.word2idx = {word: i for i,word in enumerate(keyvec.index2word)}
        self.words_dict = torch.FloatTensor(keyvec.vectors)
        self.embedding=nn.Embedding.from_pretrained(self.words_dict)

    def get_length(self):
        return self.embedding.num_embeddings
        
    def get_word2idx(self):
        return self.word2idx

    def get_index(self,word):
        return self.word2idx.get(word)

    def get_wordsdict(self):
        return self.words_dict

    def get_embedding(self):
        return self.embedding

    def get_vector(self,word):
        return self.embedding.weight[self.word2idx[word]]