# -*- coding: utf-8 -*-
import re
import itertools
import sklearn
import pandas as pd
import numpy as np
import scipy.sparse as sp
from numpy.linalg import norm
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import Birch
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift



def normalize(documents):
    for i in range(len(documents)):
        documents[i]=re.sub('।',' ।',documents[i])
        documents[i]=re.sub('\(','',documents[i])
        documents[i]=re.sub('\)','',documents[i])
        documents[i]=re.sub(',','',documents[i])
        documents[i]=re.sub('—',' — ',documents[i])
        documents[i]=re.sub(':',' :',documents[i])
        documents[i]=re.sub('‘','',documents[i])
        documents[i]=re.sub('\d+','',documents[i])
        documents[i]=re.sub(r'[?|$|.|!|@]',r'',documents[i])
        documents[i]=re.sub('-',' ',documents[i])
    
    f2= open("stopwordspun", "r")
    stopwords = [x.strip() for x in f2.readlines()]
    filt_docs = [[word for word in document.split() if word not in stopwords]for document in documents]
    
    punc=['|','।','?',',','"',']','[','.','-','()','(',')','—','/',]
    docs_filt= [[word for word in filt_doc if word not in punc]for filt_doc in filt_docs]
    
    def generate_stem_words(word):
        if word.endswith("ਆਂ"):
            if word.endswith("ੀਆਂ"):
                word = word.replace("ੀਆਂ","ੀ")
            elif word.endswith("ਿਆਂ"):
                word = word.replace("ਿਆਂ","'ੇ")
            elif word.endswith("ੂਆਂ"):
                word=word.replace("ੂਆਂ","")
            elif word.endswith("ਆਂ"):
                word = word.replace("ਆਂ","")
        if word.endswith("ਾਂ"):
            if word.endswith("ਵਾਂ"):
                word=word.replace("ਵਾਂ","")
            elif word.endswith("ਾਂ"):
                word=word.replace("ਾਂ","")
        if word.endswith("ੀਏ"):
            word=word.replace("ੀਏ","ੀ")
        if word.endswith("ਈ"):
            word=word.replace("ਈ","")
        if word.endswith("ੇ"):
            word=word.replace("ੇ","ਾ")    
        if word.endswith("ੀਓ"):
            word=word.replace("ੀਓ","ੀ")  ## ਰੇਡੀਓ , ਬਚਾਈਓ exception
        if word.endswith("ਿਓ"):
            word=word.replace("ਿਓ","ਾ")
        if word.endswith("ੋਂ"):
            word=word.replace("ੋਂ","")
        if word.endswith("ੋ"):
            word=word.replace("ੋ","")
        if word.endswith("ੀ"):
            word=word.replace("ੀ","")
        if word.endswith("ਉਣ"):
            word=word.replace("ਉਣ","")
        if word.endswith("ਿਉਂ"):
            word=word.replace("ਿਉਂ","ਾ")
        if word.endswith("ਈਆ"):
            word=word.replace("ਈਆ","ਈ")
        if word.endswith("ੀਆ"):
            word=word.replace("ੀਆ","ੀ")
        if word.endswith("ਿਆ"):
            word=word.replace("ਿਆ","ਾ")
    
        return(word)
    
    stem_doc = [[generate_stem_words(word) for word in doc]for doc in docs_filt]
    
    mylist = []

    f= open('pun_syn.txt', "r")
    contents = f.readlines()
    f.close()
    cnt=0
    for line in itertools.islice(contents, 4, 194183,6):
        mylist.append(line[18:].split(','))
        cnt=cnt+1
    mylist = [[e.strip() for e in l]for l in mylist]

    def syn_rep(word):
        for syns in mylist:
            for syn in syns:
                if syn==word:
                    return syns[0]

        return word
    
    stem_docs = [[syn_rep(word) for word in doc]for doc in stem_doc]
    stem_docs = [[word for word in document if len(word)>3]for document in stem_docs]
    filt_fin = [[word.decode('utf-8') for word in doc]for doc in stem_docs]
    return filt_fin



def sentences_to_bag_of_words(in_sentences):
    bag_of_words = {}
    for sentence in in_sentences:
        for word in sentence:
            if word in bag_of_words:
                bag_of_words[word] += 1
            else:
                bag_of_words[word] = 1
    arr=[]
    for sentence in in_sentences:
        vec=bag_of_words
        ar=[]
        for key in sorted(vec):
            vec[key]=0
        for word in sentence:
            vec[word] += 1
        for k in sorted(vec):
            ar.append(vec[k])
        arr.append(ar)
    
    arr=np.asarray(arr)
    arr = np.array(arr, dtype='float64')
    return sparse.csr_matrix(arr)



def sentences_to_tfidf(filt_fin):
    bag_of_words = {}
    for sentence in filt_fin:
        for word in sentence:
            if word in bag_of_words:
                bag_of_words[word] += 1
            else:
                bag_of_words[word] = 1
    features=[]
    df=[]
    for key in sorted(bag_of_words):
        df.append(bag_of_words[key])
        features.append(key)
    df=np.asarray(df)
    features=np.asarray(features)
    df=df+1
    total_docs = 1 + len(filt_fin)
    idf = 1.0 + np.log(float(total_docs) / df)
    tf=sentences_to_bag_of_words(filt_fin)
    bow_features = sparse.csr_matrix(tf)
    total_features = bow_features.shape[1]
    idf_diag = sp.spdiags(idf, diags=0, m=total_features, n=total_features)
    idf = idf_diag.todense()
    tfidf = tf * idf
    norms = norm(tfidf, axis=1)
    norm_tfidf = tfidf / norms[:, None]
    corpus_tfidf=sparse.csr_matrix(norm_tfidf)
    
    return bag_of_words, corpus_tfidf



def advance_vectorize(corpus, num_features=300, vec_type='avg'):
    model = KeyedVectors.load_word2vec_format('wiki.pa.vec')
    if vec_type == 'avg':
        vocabulary = set(model.index2word)

        def average_word_vectors(words, model, vocabulary, num_features):
            feature_vector = np.zeros((num_features,),dtype="float64")
            nwords = 0.
            for word in words:
                if word in vocabulary: 
                    nwords = nwords + 1.
                    feature_vector = np.add(feature_vector, model[word])
            if nwords:
                feature_vector = np.divide(feature_vector, nwords)
            return feature_vector
        
        features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                        for tokenized_sentence in corpus]
        return np.array(features)
    
    
    if vec_type == 'tfidf':
        tfidf_vocab, tdidf_features =sentences_to_tfidf(corpus)
        i=0
        for key in sorted(tfidf_vocab):
            tfidf_vocab[key] = i
            i=i+1
        tfidf_vectors = tdidf_features.todense()
        docs_tfidfs = [(doc, doc_tfidf) 
                        for doc, doc_tfidf 
                        in zip(norm_doc, tfidf_vectors)]
        
        def tfidf_wtd_avg_word_vectors(words, tfidf_vector, tfidf_vocabulary, model, num_features):
            word_tfidfs = [tfidf_vector[0, tfidf_vocabulary.get(word)] 
                           if tfidf_vocabulary.get(word) 
                           else 0 for word in words]    
            word_tfidf_map = {word:tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}
    
            feature_vector = np.zeros((num_features,),dtype="float64")
            vocabulary = set(model.index2word)
            wts = 0.
            for word in words:
                if word in vocabulary: 
                    word_vector = model[word]
                    weighted_word_vector = word_tfidf_map[word] * word_vector
                    wts = wts + 1.
                    feature_vector = np.add(feature_vector, weighted_word_vector)
            if wts:
                feature_vector = np.divide(feature_vector, wts)
            return feature_vector
        
        features = [tfidf_wtd_avg_word_vectors(words, tfidf_vector, tfidf_vocab ,model, num_features)
                    for words, tfidf_vector in docs_tfidfs]

        return np.array(features) 


