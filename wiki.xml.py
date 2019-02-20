import os.path
import sys
import jieba
import seg_sentence
 
from gensim.corpora import WikiCorpus

# python process_wiki.py zhwiki-latest-pages-articles.xml.bz2 wiki.zh.txt
if __name__ == '__main__':
    # program = os.path.basename(sys.argv[0])
 
    # check and process input arguments
    # if len(sys.argv) < 3:
    #     print globals()['__doc__'] % locals()
    #     sys.exit(1)
    # inp, outp = sys.argv[1:3]
    # space = " "
    # i = 0
 
    # output = open(outp, 'w',encoding='utf8')
    # wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    # for text in wiki.get_texts():
    #     output.write(space.join(text) + "\n")
    #     i = i + 1
    #     if i%1000==0 :
    #         print('save '+str(i))
 
    # output.close()
    # 中文分词
    wiki = open('./data/wiki.zh.txt', 'r',encoding='utf8')
    wiki_out = open('./data/wiki.zh.seg.txt','w',encoding='utf8')
    line = wiki.readline()
    lastline=""
    while line:
        sent=''
        for word in jieba.cut(seg_sentence.segment(line)):
            sent += word
        sent += '\n'
        wiki_out.write(sent)
        line = wiki.readline()
    wiki_out.close()
    wiki.close()
