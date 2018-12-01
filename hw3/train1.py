import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, PReLU, LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
# from keras.units.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

# import matplotlib.pyplot as plt
import pandas as pd
import os

val=True
_batch_size = 128


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

def flip_aug(X, Y):
    newX = np.r_[X, np.flip(X, 1)]
    newY = np.r_[Y, Y]
    return newX, newY

def val_split(X, Y):
    num = 1000
    return X[num:], Y[num:], X[:num], Y[:num]
    # return X[:4*num], Y[:4*num], X[4*num:], Y[4*num:]

def train(X, Y, valX, valY):
    model = Sequential()
    model.add(Convolution2D(
        filters = 128,
        kernel_size = (5, 5),
        input_shape = (48, 48, 1),  # height, width, channels
        padding= 'same',
        kernel_initializer = "glorot_normal"
        
    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./20))
    # can add Drop
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, (5, 5),padding= 'same',kernel_initializer = "glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./20))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    # model.add(Dropout(0.3))
    model.add(Convolution2D(512, (3, 3),padding= 'same',kernel_initializer = "glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./20))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))    
    # 
    model.add(Convolution2D(512, (3, 3),padding= 'same',kernel_initializer = "glorot_normal"))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    # model.add(Convolution2D(32, (3, 3),padding= 'same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    # model.add(Dropout(0.2))
    # model.add(Dense(units=256)) # PReLU(alpha_initializer='zeros')
    # model.add(LeakyReLU(alpha=1./20))
    # # model.add(BatchNormalization())
    # model.add(Dropout(0.25))
    # model.add(Dense(units=512))
    # model.add(LeakyReLU(alpha=1./20))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.25))
    model.add(Dense(units=512,kernel_initializer = "glorot_normal"))
    model.add(LeakyReLU(alpha=1./20))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(units=512,kernel_initializer = "glorot_normal"))
    model.add(LeakyReLU(alpha=1./20))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # 
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
    # adam = Adam(lr=0.003)
    filepath = "selfmodel3_{epoch:02d}-{val_acc:.3f}.hdf5"
    checkpoint = ModelCheckpoint(os.path.join('save_model', filepath) , monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if valX is not None:
        model.fit_generator(datagen.flow(X, Y, batch_size=_batch_size), steps_per_epoch=(10*len(X)/_batch_size) , epochs=1, validation_data=(valX, valY)) # , callbacks= [checkpoint]
    else:
        model.fit_generator(datagen.flow(X, Y, batch_size=_batch_size), steps_per_epoch=(10*len(X)/_batch_size) , epochs=258)
    return model

trainY, feature = load_data('train.csv')
trainX = feature2img(feature)
# trainX, trainY = flip_aug(trainX, trainY)

datagen = ImageDataGenerator(
    rotation_range=35,
    # zoom_range=0.2,
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip=True,
    # vertical_flip=True,
    shear_range=0.25,
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
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
np.savetxt('submission.csv', output, delimiter=",", fmt='%s'+ ',%s', header='id'+',label', comments='')
# plt.imshow(img_set[2])
# plt.show()
