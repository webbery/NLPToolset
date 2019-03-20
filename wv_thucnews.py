# 使用thucnews数据集训练词向量

import os
import jieba
import seg_sentence
import pickle

def getFilePathList(rootDir):
    filePath_list = []
    for walk in os.walk(rootDir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list

filePath_list = getFilePathList('../data/THUCNews')
# 标签
# label_list = []
# for filePath in filePath_list:
#     label = filePath.split('\\')[1]
#     label_list.append(label)

# #保存标签
# with open('../data/label_list.pkl', 'wb') as file:
#     pickle.dump(label_list, file)

#保存文本内容
content_list = []
for filePath in filePath_list:
    file = open(filePath, 'r',encoding='utf8')
    line = file.readline()
    sentence = ''
    while line:
        sentence+=line
        line = file.readline()
    content_list.append(sentence)
with open('../data/content_list.pkl', 'wb') as file:
    pickle.dump(content_list, file)

# gensim_out = open('../data/zh.seg.txt','w',encoding='utf8')

# def process(corpus):
#     line = corpus.readline()
#     while line:
#         sent=''
#         for word in jieba.cut(seg_sentence.segment(line)):
#             sent+=word
#         sent += '\n'
#         gensim_out.write(sent)
#         line = corpus.readline()

#循环处理每个文本,获取分词
# for filePath in filePath_list:
#     primitive_corpus = open(filePath, 'r',encoding='utf8')
#     process(primitive_corpus)

# gensim_out.close()