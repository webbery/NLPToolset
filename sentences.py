# 打开文件，每次读取一行，并做分词，支持逆序
import seg_sentence

class Sentences():
    def __init__(self,filename):
        self.input_file = open(filename, 'r',encoding='utf8')
    
    def readline(self,reverse=False):
        line = self.input_file.readline()
        while line:
            #向量化
            words = seg_sentence.segment(line,type='arr')
            #句子按词翻转逆序
            if reverse==True:
                words.reverse()
            yield words
            line = self.input_file.readline()
        return None