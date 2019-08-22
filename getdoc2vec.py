from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import jieba
import  numpy as np
import pandas as pd
import codecs
import pickle
from scipy import sparse
from sklearn import preprocessing
import read_write
from scipy.sparse import vstack

#类tfidf()形成词向量矩阵
class GetDoc2vec():

    def __init__(self):
        self.stopwords = codecs.open('D:/xfdata/stop_words.txt', 'r', encoding='GBK').readlines()
        self.stopwordlist = [w.strip() for w in self.stopwords]

    def tokenization(self, doc):
        words = jieba.cut(doc)
        result = " ".join(words)
        return result

    def tomatrix(self, docs):
        dockeys = []
        if len(docs)>1:
            for row in docs.itertuples():
                print(row)
                soup = getattr(row, 'GKXX')
                word_list = self.tokenization(soup)
                dockeys.append(word_list)
            counter = CountVectorizer(stop_words = self.stopwordlist) #countvectorizer词汇表，有多少个，词向量就是多少维度
            counts = counter.fit_transform(dockeys)
            read_write.tosave("D:/xftest/forir/CountVectorizer.pkl",counts) #保存词向量
            get_feature=counter.get_feature_names()
            read_write.tosave("D:/xftest/forir/get_feature.pkl",get_feature) #保存特征属性
            tfidfer = TfidfTransformer()
            tfidf = tfidfer.fit_transform(counts)
            read_write.tosave('D:/xftest/forir/doc2vec0.pkl',tfidf) #保存tfidf向量
            return tfidf
        else:   #将一条文档向量化
            word_list = self.tokenization(docs[0])
            dockeys.append(word_list)
            get_feature = read_write.toload("D:/xftest/forir/get_feature.pkl")
            counter=CountVectorizer(stop_words = self.stopwordlist,vocabulary=get_feature)
            counts = counter.transform(dockeys)
            oldcounts=read_write.toload("D:/xftest/forir/CountVectorizer.pkl")
            countsvec=vstack((oldcounts,counts))
            tfidfer = TfidfTransformer()
            tfidf = tfidfer.fit_transform(countsvec)
            return tfidf

    '''
if __name__ == "__main__":
    df = pd.read_csv('D:/xftest/gkxx.csv',',',encoding='ansi',header=0)
    doc1 = pd.DataFrame(df['GKXX']).astype(str)
    qdoc=['安置房已拿，无钱装修，要求帮其解决。']
    queryvec = GetDoc2vec()
    q=queryvec.tomatrix(qdoc)
    doc2vec = read_write.toload('D:/xftest/forir/doc2vec.pkl')
    print(doc2vec.shape)
    print(q.dot(doc2vec[1].T))
    '''











