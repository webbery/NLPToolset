# https://storage.googleapis.com/nlp_chinese_corpus/webtext2019zh.zip 数据预处理

# 该py提取问题的描述和回复的内容，并做分词生成gensim训练词向量需要的语料库

# 数据集的数据结构
# {"qid":<qid>,"title":<title>,"desc":<desc>,"topic":<topic>,"star":<star>,"content":<content>,
# "answer_id":<answer_id>,"answerer_tags":<answerer_tags>}
# 其中，qid是问题的id，title是问题的标题，desc是问题的描述，可以为空；topic是问题所属的话题，star是该回复的点赞个数，
# content是回复的内容，answer_id是回复的ID,answerer_tags是回复者所携带的标签


import os.path
import sys
import jieba
import seg_sentence
import json

def process(corpus):
    line = corpus.readline()
    jstr = json.loads(line)
    return jstr['title']+" "+jstr['content']

primitive_corpus = open('./data/web_text_zh_train.json', 'r',encoding='utf8')

gensim_out = open('./data/zh.seg.txt','w',encoding='utf8')
line = process(primitive_corpus)
# lastline=""
i=0
while line:
    sent=''
    for word in jieba.cut(seg_sentence.segment(line)):
        sent += word
    sent += '\n'
    gensim_out.write(sent)
    line = process(primitive_corpus)
    if i%1000==0:
        print(sent)
    i+=1
gensim_out.close()
primitive_corpus.close()