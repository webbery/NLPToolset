# 分割中文句子
import jieba

def is_chinese(uchar):
    # 判断一个unicode是否是汉字
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False
    
def is_number(uchar):
    # 判断一个unicode是否是数字
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False

def is_alphabet(uchar):
    # 判断一个unicode是否是英文字母
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False

def is_legal(uchar):
    # 判断是否非汉字，数字和英文字符
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return False
    else:
        return True

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