import numpy as np
import pandas as pd
import csv
import jieba
import jieba.analyse
import os
from keras.layers import Input, Dense, Activation, Embedding, LSTM, Bidirectional, Dropout, BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from gensim.models import Word2Vec
from keras.models import load_model
import pickle
import sys
# import matplotlib.pyplot as plt

maxlen = 250
_batch_size=128
_epochs= 25

jieba.set_dictionary(sys.argv[4])

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

def load_stopwords(path):
    print('start loading stopwords')
    data = []
    # sentence = []
    with open(path, newline='\n', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=",")
        for i, r in enumerate(reader):
            data.append(r[0])
                
    data = np.array(data)
    # print(data[:10])
    return data

def word_segmentation(trainX):
    print('start word_segmentation')
    seg_data = []
    # trainX.astype("UTF-8")
    for x in trainX:
        sentence = x[0]
        seg = jieba.lcut(sentence, cut_all=True)
        # seg_r = []
        # for s in seg:
        #     if s not in stop_words:
        #         seg_r.append(s)
        # print(seg_r)
        seg_data.append(seg)
    print('seg_data', seg_data[:10])
    return seg_data

def word_vec(trainX):
    print('w2v')
    w2v_model = Word2Vec(trainX, size=maxlen, window=5, min_count=3, workers=8, iter=50)
    # w2v_model.save("word2vec8.model")
    # w2v_model = Word2Vec.load("word2vec.model")

    embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
    print("embedding_matrix:", embedding_matrix.shape)
    word2idx = {}
    vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
    for i, vocab in enumerate(vocab_list):
        # print(i)
        word, vec = vocab
        # if i < 10:
        #     print(word)
        embedding_matrix[i + 1] = vec
        word2idx[word] = i + 1
    
    # f = open("word2idx8.pkl", "wb")
    # pickle.dump(word2idx, f)
    
    return embedding_matrix, word2idx

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

# def embedding_matrix(w2v_model, trainX):
#     matrix = np.zeros((len(trainX) + 1, w2v_model.vector_size)
#     for 
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
def validation_split(trainX, trainY):
    valX = trainX[:1000]
    valY = trainY[:1000]
    train_X = trainX[1000:]
    train_Y = trainY[1000:]
    return train_X, train_Y, valX, valY

def train(trainX, trainY, embedding_matrix, valX, valY):
    main_input = Input((trainX.shape[1],))
    word_embedding = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1]
    , input_length=maxlen, weights=[embedding_matrix], trainable=False)(main_input)

    training_data = Bidirectional(LSTM(64))(word_embedding)
    training_data = Dropout(0.15)(training_data)
    training_data = Bidirectional(LSTM(64))(word_embedding)
    training_data = Dropout(0.5)(training_data)

    training_data = Dense(64, activation='relu')(training_data)
    training_data = BatchNormalization()(training_data)
    training_data = Dropout(0.2)(training_data)
    training_data = Dense(64, activation='relu')(training_data)
    training_data = BatchNormalization()(training_data)
    training_data = Dropout(0.2)(training_data)
    # training_data = Dense(64, activation='relu')(training_data)
    # training_data = BatchNormalization()(training_data)
    # training_data = Dropout(0.2)(training_data)
    # model.add(Dense(units=64, activation='relu'))

    # # model.add(BatchNormalization())
    # # model.add(Dropout(0.2))
    # model.add(Dense(units=128, activation='relu'))
    # # # model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    main_output = Dense(1, activation='sigmoid')(training_data)

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    # checkpoint = ModelCheckpoint("", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    # filepath = "model8_{epoch:02d}-{val_acc:.3f}.hdf5"
    # checkpoint = ModelCheckpoint(os.path.join('save_model', filepath) , monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
    model.summary()
    history = model.fit(trainX, trainY, batch_size=_batch_size, epochs=_epochs, validation_data=(valX, valY))
    return model, history

# def plot_history(history):
#     # list all data in history
#     print(history.history.keys())
#     # summarize history for accuracy
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()
#     # summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()   

trainX = load_data(sys.argv[1])
trainY = load_label(sys.argv[2])
testX = load_data(sys.argv[3])
# stop_words = load_stopwords("stopWords.txt")
trainX_seg = word_segmentation(trainX)
testX_seg = word_segmentation(testX)
data = trainX_seg+testX_seg
embedding_matrix, word2idx = word_vec(data)
# print('em:', embedding_matrix[:10])
# print('w2i:', word2idx[:10])
X = text_to_index(trainX_seg, word2idx)
X = pad_sequences(X, maxlen)
# print('X',X[0])
train_X, train_Y, val_X, val_Y = validation_split(X, trainY)
# print(train_X.shape)

model, history = train(train_X, train_Y, embedding_matrix, val_X, val_Y)

# plot_history(history)

# model = load_model("save_model/model_16-0.751.hdf5")
# model = load_model("save_model/model2_10-0.757.hdf5")
# print('start testing--------------')
# # testX = load_data("test_x.csv")
# # print(testX.shape)
# # testX_seg = word_segmentation(testX)
# # testX_embedding_matrix, testX_word2idx = word_vec(testX_seg)
# test_X = text_to_index(testX_seg, word2idx)
# print('start padding--------------')
# test_X = pad_sequences(test_X, maxlen)
# print('start predict--------------')
# y_predict = model.predict(test_X)
# print('before argmax', y_predict[:10])
# y_output = np.zeros((y_predict.shape))
# y_output[np.where(y_predict >= 0.5)] = 1
# print('argmaxed', y_output[:10])
# id = [x for x in range(len(y_output))]
# id = np.array(id)
# output = np.c_[id.astype(int), y_output.astype(int)]
# np.savetxt('submission.csv', output, delimiter=",", fmt='%s'+ ',%s', header='id'+',label', comments='')