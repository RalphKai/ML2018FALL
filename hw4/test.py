import numpy as np
import pandas as pd
import csv
import jieba
import jieba.analyse
import os
from keras.layers import Input, Dense, Activation, Embedding, LSTM, Bidirectional, Dropout
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from gensim.models import Word2Vec
from keras.models import load_model
import pickle
import sys

maxlen = 250
_batch_size=512
_epochs=25

jieba.set_dictionary(sys.argv[2])

def load_data(path):
    print('start loading')
    data = []
    # sentence = []
    with open(path, newline='\n', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=",")
        for i, r in enumerate(reader):
            if i != 0:
                data.append(r[1])
    data = np.array(data)
    # print(data[:10])
    return data.reshape((len(data), 1))

def load_label(path):
    trainY = pd.read_csv(path)
    trainY = trainY['label'].values
    return trainY.reshape((trainY.shape[0],1))

def word_segmentation(trainX):
    print('start word_segmentation')
    seg_data = []
    # trainX.astype("UTF-8")
    for x in trainX:
        seg = jieba.lcut(x[0], cut_all=True)
        seg_data.append(seg)
    return seg_data

def text_to_index(corpus, word2idx):
    print('start text_to_index')
    new_corpus = []
    for doc in corpus:
        # print(doc)
        new_doc = []
        for word in doc:
            # print(word)
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        new_corpus.append(new_doc)
    return np.array(new_corpus)

def split_word(X):
    new_X = []
    for x in X:
        for t in x:
            # print(t)
            nx = list(t)
            # print(nx)
        new_X.append(nx)
    return new_X

print('start testing--------------')
testX = load_data(sys.argv[1])
# testX = [["在說別人白痴之前，先想想自己"],["再說別人之前先想想自己，白痴"]]
print(testX.shape)
testX_seg = word_segmentation(testX)
testX_seg2 = split_word(testX)
# testX_embedding_matrix, testX_word2idx = word_vec(testX_seg)
pkl_file = open("word2idx.pkl", "rb")
word2idx = pickle.load(pkl_file)
pkl_file.close()

pkl_file1 = open("word2idx1.pkl", "rb")
word2idx1 = pickle.load(pkl_file1)
pkl_file1.close()

pkl_file2 = open("word2idx2.pkl", "rb")
word2idx2 = pickle.load(pkl_file2)
pkl_file2.close()

test_X = text_to_index(testX_seg, word2idx)
test_X1 = text_to_index(testX_seg, word2idx1)
test_X2 = text_to_index(testX_seg2, word2idx2)
print('start padding--------------')
test_X = pad_sequences(test_X, maxlen)
test_X1 = pad_sequences(test_X1, maxlen)
test_X2 = pad_sequences(test_X2, maxlen)
model = load_model("model.hdf5")
model1 = load_model("model1.hdf5")
model2 = load_model("model2.hdf5")

print('start predict--------------')
y_predict1 = model.predict(test_X, batch_size=128)
y_predict2 = model1.predict(test_X1, batch_size=128)
y_predict3 = model2.predict(test_X2, batch_size=128)
# print('before argmax', y_predict1[:10], y_predict2[:10])
y_predict = (y_predict1 + y_predict2 + y_predict3) / 3
# y_predict = y_predict1
y_output = np.zeros((y_predict.shape))
y_output[np.where(y_predict >= 0.5)] = 1
print('argmaxed', y_output[:10])
id = [x for x in range(len(y_output))]
id = np.array(id)
output = np.c_[id.astype(int), y_output.astype(int)]

np.savetxt(sys.argv[3], output, delimiter=",", fmt='%s'+ ',%s', header='id'+',label', comments='')