from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec_corpusfile
from gensim.models import KeyedVectors
import codecs
import os

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
                
# tcnn=TextCNN()
# weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
# embedding = nn.Embedding.from_pretrained(weight)
# input = torch.LongTensor([1])
# val = embedding(input)
# print(val)

# wf = codecs.open('./data/wiki.zh.word.2.text','w','utf-8')
# f = codecs.open('./data/wiki.zh.word.text', 'r','utf-8')
# line = f.readline()
# lastline=""
# while line:
#     try:
#         if lastline!=line:
#             wf.write(line)
#             lastline = line
#         line = f.readline()
#     except UnicodeDecodeError:
#         print('error')
#         f.seek(2,1)
# f.close()
# wf.close()

# sentences = LineSentence('./data/wiki.zh.word.2.text')
# model  = Word2Vec(sentences,size = 400,hs = 1,window =3)
# model.save('./data/wiki.zh.400.model')
# print(model.most_similar('科学家'))
# i=0
model2 = Word2Vec.load('./data/wiki.zh.400.model')
print('load finish')
# model2.build_vocab(corpus_file='./data/min.test.txt', update=True)
# print(model2.corpus_total_words)
model2.build_vocab(corpus_file='./data/zh.seg.txt', update=True)
print('build finish')
model2.train(corpus_file='./data/zh.seg.txt',total_examples=model2.corpus_count,epochs=model2.iter,total_words=model2.corpus_total_words)
# model2.train(corpus_file='./data/min.test.txt',total_examples=model2.corpus_count,epochs=model2.iter,total_words=model2.corpus_total_words)
print('train finish')
# wiki = open('./data/wiki.zh.seg.txt','r',encoding='utf8')
# line = wiki.readline()
# while line:
#     more_sentences = [[line]]
#     model2.build_vocab(more_sentences, update=True)
#     model2.train(more_sentences,total_examples=model2.corpus_count,epochs=model2.epochs)
#     line = wiki.readline()
#     i+=1
#     if i%1000==0:
#         print('sentence '+str(i))
# wiki.close()
model2.save('./data/g2.400.model')
print(model2.most_similar('科学家'))