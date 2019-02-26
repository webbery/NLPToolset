# 分割中文句子
import jieba

def segment(sentence,type="str"):
    sentence_seged = jieba.cut(sentence.strip()) 
    stopwords = [",","，","。","?","？","（","）","(",")","《","》","、","/","！","「","」","“","”","：","<",">","－","／"," "]
    if type=="str":
        outstr = ''
        for word in sentence_seged:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr
    else:
        outstr = []
        for word in sentence_seged:
            if word not in stopwords:
                if word != '\t':
                    outstr.append(word)
        return outstr