import os.path
import sys
 
from gensim.corpora import WikiCorpus
# python process_wiki.py zhwiki-latest-pages-articles.xml.bz2 wiki.zh.txt
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
 
    # check and process input arguments
    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = " "
    i = 0
 
    output = open(outp, 'w')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        i = i + 1
 
    output.close()