import numpy as np
from keras.models import load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, PReLU, LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
# from keras.utils import plot_model

# import matplotlib.pyplot as plt
import pandas as pd
import os

def load_data(data_name):
    data = pd.read_csv(data_name)
    D = data['feature'].str.split(' ', n=48*48, expand=True)
    print(data.head())
    data = data.values
    D = D.values
    label = data[:, 0]
    feature = D
    print(feature.shape)
    return np.array(label), np.array(feature)

def data_preprocess(label):
    y = np.zeros((len(label), 7))
    for n in range(len(label)):
        y[n, label[n]] = 1
    return y

def output_process(y_hat):
    y = np.zeros((len(y_hat), 1))
    for n in range(y_hat.shape[0]):
        for d in range(y_hat.shape[1]):
            if y_hat[n, d] == 1:
                y[n] = d
    return y

def feature2img(feature):
    for i in range(feature.shape[0]):
        feature[i] = np.array(feature[i]).astype(np.float)
        feature[i] = feature[i]
    feature = feature.reshape((feature.shape[0], 48, 48, 1))
    feature = feature / 255
    return feature


id1, test_X = load_data('test.csv')
test_X = feature2img(test_X)
model = load_model("save_model/selfmodel2_367-0.652.hdf5")
y_predict = model.predict_classes(test_X)
# y_predict = output_process(y_predict)

id = [x for x in range(len(y_predict))]
id = np.array(id)
output = np.c_[id.astype(int), y_predict.astype(int)]

print(model.summary())
np.savetxt('submission.csv', output, delimiter=",", fmt='%s'+ ',%s', header='id'+',label', comments='')
