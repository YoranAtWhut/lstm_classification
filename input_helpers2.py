# -*- coding: utf-8 -*-

import numpy as np
import gc
import random
import jieba.posseg as pseg

class InputHelper(object):
    #分词
    def cut_sentence(self,sent):
        max_len = 20
        
        words = []
        _words = pseg.cut(sent)
        for _word in _words:
            words.append(_word.word)
            
        if len(words)>max_len:
            words = words[:20]
        else:
            words = words + ['null'] * (20-len(words))
        
        return words
    
    def load_cn_data_from_files(self,classify_files):
        data = []
        labels = []
        
        with open(classify_files,'r') as df:
            for s in df:
                s = s.strip()
                ss = s.split(',')
                s1 = self.cut_sentence(ss[0])
                data.append(s1)
                label = int(ss[1])
                labels.append(label)
        
        return data,labels
    
    def getData(self,filepath):
        print('loading training data from:' + filepath)
        data,labels = self.load_cn_data_from_files(filepath)
        alllabels = set(labels)
        return np.asarray(data),np.asarray(labels)
    
    def batch_iter(self,data,batch_size,num_epochs,shuffle=True):
        '''
        generate a batch iterator for a dataset
        '''
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            # shuffle the data at each epoch
            if shuffle:
                shuffle_indices = list(range(data_size))
                random.shuffle(shuffle_indices)
                shuffled_data = []
                for shuffle_indice in shuffle_indices:
                    shuffled_data.append(data[shuffle_indice])
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
    
    def getDataSets(self,training_paths,percent_dev,batch_size):
        x,y = self.getData(training_paths)
        sum_no_of_batches = 0
        
        #randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_idx = -1*len(y_shuffled)*percent_dev//100
        
        #split train/test set
        x_train,x_dev = x_shuffled[:dev_idx],x_shuffled[dev_idx:]
        y_train,y_dev = y_shuffled[:dev_idx],y_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))
        sum_no_of_batches = sum_no_of_batches + (len(y_train)//batch_size)
        train_set = (x_train,y_train)
        dev_set = (x_dev,y_dev)
        gc.collect()
        return train_set,dev_set,sum_no_of_batches
        
                


if __name__ == '__main__':
    a = InputHelper()
    #data,labels = a.get_data('data.csv')
    train_set,dev_set,sum_no_of_batches = a.getDataSets('data.csv',20,10)
    batches = a.batch_iter(list(zip(train_set[0],train_set[1])),10,
                                       2)