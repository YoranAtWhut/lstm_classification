# -*- coding: utf-8 -*-

#import data_helper as wv
from gensim.models import word2vec
import jieba.posseg as pseg
import logging
import numpy as np

stop_word = [' ','，','。']

def get_raw_sentences(filepath):
    file = open(filepath)
    sentences = []
    for line in file.readlines():
        s = process(line.strip())
        sentences.append(s)
    return sentences

def process(sent):
    words = []
    _words = pseg.cut(sent)
    for _word in _words:
        if _word.word not in stop_word:
            words.append(_word.word)
    return words
    
def set_model(filepath,min_count=1):
    sentences = get_raw_sentences(filepath)
    model = word2vec.Word2Vec(sentences,min_count=min_count)
    model.save(filepath[:-4]+'.model')
    return model
    
def load_model():
    model = word2vec.Word2Vec.load('embedding_train.model')
    return model

def embedding_lookup(batch_size,length,data):
    model = load_model()
    embedding = []
    for i,sentence in enumerate(data):
        sentence_embedding = []
        for j,word in enumerate(sentence.tolist()):            
            if word == 'null':
                sentence_embedding.append([0.]*100)
            else:
                sentence_embedding.append(list(model[word]))
        embedding.append(sentence_embedding)
    return embedding
            

if __name__ == '__main__':
    model = set_model('embedding_train.txt')
    model = load_model()