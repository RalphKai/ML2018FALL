import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# import matplotlib.pyplot as plt
import pandas as pd

val=True
_batch_size = 32


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
    feature = feature.astype(float)
    feature = feature / 255
    return feature

def val_split(X, Y):
    num = int(len(X) * 0.2)
    return X[:4*num], Y[:4*num], X[4*num:], Y[4*num:]

def train(X, Y, valX, valY):
    model = Sequential()
    model.add(Convolution2D(
        filters = 6,
        kernel_size = (2, 2),
        input_shape = (48, 48, 1),  # height, width, channels
        padding= 'valid'

    ))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.3))
    model.add(Convolution2D(4, (2, 2),padding= 'valid'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2, 2)))
    # 
    # model.add(Convolution2D(32, (3, 3),padding= 'valid'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D((2, 2)))
    # # # model.add(Dropout(0.3))
    # model.add(Convolution2D(32, (3, 3),padding= 'same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(units=16, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(units=16, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # model.add(Dense(units=64, activation='relu'))
    # # model.add(BatchNormalization())
    # # model.add(Dropout(0.2))
    # model.add(Dense(units=64, activation='relu'))
    # # model.add(BatchNormalization())
    # # model.add(Dropout(0.2))
    # model.add(Dense(units=128, activation='relu'))
    # # # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(units=7, activation='softmax'))

    # model = Sequential([
    # Convolution2D(64, (3, 3), input_shape = (48, 48, 1), padding='same',
    #        activation='relu'),
    # Convolution2D(64, (3, 3), activation='relu', padding='same'),
    # MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    # Convolution2D(128, (3, 3), activation='relu', padding='same'),
    # Convolution2D(128, (3, 3), activation='relu', padding='same',),
    # MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    # Convolution2D(256, (3, 3), activation='relu', padding='same',),
    # Convolution2D(256, (3, 3), activation='relu', padding='same',),
    # Convolution2D(256, (3, 3), activation='relu', padding='same',),
    # MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    # Convolution2D(512, (3, 3), activation='relu', padding='same',),
    # Convolution2D(512, (3, 3), activation='relu', padding='same',),
    # Convolution2D(512, (3, 3), activation='relu', padding='same',),
    # MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    # Dropout(0.5),

    # Flatten(),
    # Dense(128, activation='relu'),
    # Dropout(0.2),
    # Dense(256, activation='relu'),
    # Dropout(0.2),
    # Dense(256, activation='relu'),
    # Dropout(0.2),
    # Dense(7, activation='softmax')
    # ])

    # checkpoint = ModelCheckpoint("", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    adam = Adam(lr=0.0008)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    if valX is not None:
        model.fit_generator(datagen.flow(X, Y, batch_size=_batch_size), steps_per_epoch=(5*len(X)/_batch_size) , epochs=100, 
        validation_data=(valX, valY))
    else:
        model.fit_generator(datagen.flow(X, Y, batch_size=_batch_size), steps_per_epoch=(5*len(X)/_batch_size) , epochs=100)
    return model

trainY, feature = load_data('train.csv')
trainX = feature2img(feature)

datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
)

valX = None
valY_p = None

datagen.fit(trainX)
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
