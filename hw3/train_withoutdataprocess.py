import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, AveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import datetime

val=True
_epoch = 52
# plot confusion matrix ---------------------------------

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


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

def flip_aug(X, Y):
    newX = np.r_[X, np.flip(X, 1)]
    newY = np.r_[Y, Y]
    return newX, newY

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

def train(X, Y, val):
    
    model = Sequential()
    model.add(Convolution2D(
        filters = 50,
        kernel_size = (3, 3),
        input_shape = (48, 48, 1),  # height, width, channels
        padding= 'same'

    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(50, (3, 3),padding= 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    # model.add(Convolution2D(60, (3, 3),padding= 'same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(50, (3, 3),padding= 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(units=300, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=250, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=200, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=7, activation='softmax'))
    # checkpoint = ModelCheckpoint("save_without_generator/weights-improvement-{epoch:02d}-{val_acc:.3f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if val:
        model.fit(X, Y, batch_size=128, epochs=60, validation_split=0.2) # callbacks=callbacks_list,
    else:
        model.fit(X, Y, batch_size=128, epochs=_epoch)

    return model

trainY, feature = load_data('train.csv')
trainX = feature2img(feature)
# trainX, trainY = flip_aug(trainX, trainY)
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
    # valY_p = data_preprocess(trainY)
    valY_p = data_preprocess(valY)

trainY_dense = data_preprocess(trainY)
# print('training X:', trainX.shape, 'training Y:', trainY.shape)
model = train(trainX, trainY_dense, val)

y_val_predict = model.predict_classes(valX)
print(y_val_predict, valY)
cnf_matrix = confusion_matrix(valY.astype(int), y_val_predict.astype(int))

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


id1, test_X = load_data('test.csv')
test_X = feature2img(test_X)
y_predict = model.predict_classes(test_X)
# y_predict = output_process(y_predict)

id = [x for x in range(len(y_predict))]
id = np.array(id)
output = np.c_[id.astype(int), y_predict.astype(int)]

# print(model.summary())
np.savetxt('submission.csv', output, delimiter=",", fmt='%s'+ ',%s', header='id'+',label', comments='')

now = datetime.datetime.now()
otherStyleTime = now.strftime("%Y-%m-%d_%H:%M:%S")

model.save('save_withoutdataprocess_'+str(_epoch)+'_'+str(otherStyleTime)+'_'+'.h5')
# plt.imshow(img_set[2])
# plt.show()
