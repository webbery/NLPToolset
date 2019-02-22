# 分割中文句子
import jieba

def segment(sentence):
    sentence_seged = jieba.cut(sentence.strip()) 
    stopwords = [",","，","。","?","？","（","）","(",")","《","》","、","/","！","「","」","“","”","：","<",">","－","／"]
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr