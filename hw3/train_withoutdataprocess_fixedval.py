import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, AveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# import matplotlib.pyplot as plt
import pandas as pd

val=True

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

def val_split(X, Y):
    num = int(len(X) * 0.2)
    return X[:4*num], Y[:4*num], X[4*num:], Y[4*num:]

def train(X, Y, valX, valY):
    model = Sequential()
    model.add(Convolution2D(
        filters = 25,
        kernel_size = (3, 3),
        input_shape = (48, 48, 1),  # height, width, channels
        padding= 'same'

    ))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Convolution2D(50, (3, 3),padding= 'same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Convolution2D(20, (3, 3),padding= 'same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(units=250, activation='relu'))
    model.add(Dense(units=250, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=150, activation='relu'))
    model.add(Dense(units=150, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=50, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(units=7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if valX is not None:
        model.fit(X, Y, batch_size=500, epochs=40, validation_data=(valX, valY))
    else:
        model.fit(X, Y, batch_size=500, epochs=39)

    return model

trainY, feature = load_data('train.csv')
trainX = feature2img(feature)

# datagen = ImageDataGenerator(
#     zca_whitening=False,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')

# datagen.fit(trainX)


valX = None
valY_p = None
if val:
    trainX, trainY, valX, valY = val_split(trainX, trainY)
    valY_p = data_preprocess(valY)

trainY = data_preprocess(trainY)
print('training X:', trainX.shape, 'training Y:', trainY.shape)
model = train(trainX, trainY, valX, valY_p)

if val:
    y_val = model.predict_classes(valX)
    y_val = y_val.astype(int)
    Eval = 0
    for i in range(len(valX)):
        if y_val[i] != valY[i]:
            Eval = Eval + 1
    Eval = Eval / (len(valX))
    print('Eval:', Eval, '-----------')


id1, test_X = load_data('test.csv')
test_X = feature2img(test_X)
y_predict = model.predict_classes(test_X)
# y_predict = output_process(y_predict)

id = [x for x in range(len(y_predict))]
id = np.array(id)
output = np.c_[id.astype(int), y_predict.astype(int)]

print(model.summary())
np.savetxt('submission.csv', output, delimiter=",", fmt='%s'+ ',%s', header='id'+',label', comments='')
# plt.imshow(img_set[2])
# plt.show()
