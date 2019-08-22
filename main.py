from getdoc2vec import GetDoc2vec
from randomcube import RandomCube
from KNNGraph import KNNG
import pandas as pd
import read_write
import time

def get_nbit(docvec,kbest):
    l = docvec.shape[0]/kbest
    nbit = 0
    while True:
        if 2**nbit >= l:
            return nbit-1
        else:
            nbit += 1

def rukou(k_best,qdoc,doc):
    queryvec = GetDoc2vec()
    newdocvec = queryvec.tomatrix(qdoc)
    q = newdocvec[-1:]
    doc2vec = newdocvec[:-1]
    n_bit = get_nbit(doc2vec, k_best)
    rdc = RandomCube(n_bitcount = n_bit, k_dim = doc2vec.shape[1])
    rdc.inbucket(doc2vec)
    start_docsid = rdc.querydoc(q)
    knng = KNNG(n_docs = doc2vec.shape[0], k_best = k_best)
    similarid = knng.search(q, start_docsid, doc2vec)
    print(doc[similarid])


if __name__ == "__main__":
    stime=time.time()
    q=['自家原有大米加工厂，早已灭失，现在在原地建彩钢瓦约15平方，多次去信访部门要求房屋赔偿']
    rukou(k_best =10,qdoc = q,\
          doc = pd.read_csv('D:/xftest/gkxx.csv', ',', encoding = 'ansi', header = 0)['GKXX'].astype(str))
    print(time.time()-stime)

