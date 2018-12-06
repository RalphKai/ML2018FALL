import numpy as np
import pandas as pd
import csv
import jieba
import jieba.analyse

from keras.layers import Input, Dense, Activation, Embedding, LSTM, Bidirectional, Dropout
from keras.models import Model

maxlen = 100
_batch_size=32 
_epochs=20

jieba.set_dictionary('dict.txt.big')

def load_data(path):
    data = []
    # sentence = []
    with open(path, newline='\n', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=",")
        for r in reader:
            num = r[0]
            sentence = r[1]
            data.append([num, sentence])
    data = np.array(data)
    return data[1:,1].reshape((len(data[1:]), 1))

def load_label(path):
    trainY = pd.read_csv(path)
    trainY = trainY['label'].values
    return trainY.reshape((trainY.shape[0],1))

def word_segmentation(trainX):
    seg_data = []
    # trainX.astype("UTF-8")
    for x in trainX:
        seg = jieba.lcut(x[0], cut_all=True)
        seg_data.append(seg)
    return seg_data

# def equal_length(trainX):
    
    # for x in trainX:

# def keyword(X):
#     key_data = []
#     # trainX.astype("UTF-8")
#     for x in trainX[:5]:
#         # print(x)
#         seg = jieba.analyse.cut(x[0], cut_all=True)
#         seg_list = ' '.join(seg)
#         key_data.append(seg_list)
#     return np.array(key_data)

def train(trainX, trainY):
    main_input = Input(shape=(100,))
    word_embedding = Embedding(len(trainX), 64, input_length=maxlen)(main_input)

    training_data = Bidirectional(LSTM(128))(word_embedding)
    training_data = Dropout(0.5)(training_data)

    training_data = Dense(64, activation='relu')(training_data)
    training_data = Dense(64, activation='relu')(training_data)

    main_output = Dense(1, activation='sigmoid')(training_data)

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    model.fit(trainX, trainY, batch_size=_batch_size, epochs=_epochs)

    model.summary()

trainX = load_data("train_x.csv")
trainY = load_label("train_y.csv")

trainX_seg = word_segmentation(trainX)
trainX_seg = np.array(trainX_seg)
# max_l = 0
# for x in trainX_seg:
#     l = len(x)
#     if l > max_l:
#         max_l = l
# print (max_l)
l = trainX_seg[0:5]
print("l:", l)
train(trainX_seg, trainY)
