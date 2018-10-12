import numpy as np
import pandas as pd


lr = 0.001
_iteration = 6617
val = False
_lambda = 0.05

def load_training_data(filename):
    training_data = pd.read_csv(filename, encoding="ISO-8859-1")
    training_data = training_data.values
    training_data = training_data[:, 3:]
    row, col = training_data.shape
    print(training_data.shape)

    newtrain = np.zeros((18, 24 * 240))         # (feature_type, hr * days)

    count1 = 0
    for i in range(row):
        flag = i % 18
        # if flag == 9:
        #     # print(i)
        #     for j in range(col):
        #         if training_data[i, j] == '0' and j != 0:
        #             training_data[i, j] = training_data[i, j - 1]  # not use on
        #             # print(i,j)
        # if flag == 8:
        #     for j in range(col):
        #         if training_data[i, j] == '0' and j != 0:
        #             training_data[i, j] = training_data[i, j - 1] # not use on
        #             # print(i,j)

        for j in range(col):  # index = 0 : 24
            if training_data[i, j] == 'NR':
                training_data[i, j] = 0

            newtrain[flag, j + count1] = training_data[i, j]
        if flag == 17:
            count1 += 24

    cov_matrix = np.corrcoef(newtrain)
    # newtrain = np.delete(newtrain, np.arange(5688, 5761), 1)

    newtrain[np.where(newtrain < 0)] = 0
    new_row, new_col = newtrain.shape
    np.savetxt('newtrainingdata.csv', newtrain, delimiter=",", fmt='%s' + ',%s'*(new_col -1), comments='')
    # feature = np.zeros((new_col - 9 * 12, 9 * 18))
    feature = np.zeros((new_col - 9 * 12, 9 * 12))  # aware of only have 20 days
    y = np.zeros((new_col - 9 * 12,))


    count2 = 0
    for i in range(new_col - 9):  # split data to feature and label
        # print(i, i % 480)
        if i % 480 < 471:

            temp1 = newtrain[1:4, i:i + 9]
            temp2 = newtrain[5:7, i:i + 9]
            temp3 = newtrain[8:10, i:i + 9]
            temp5 = newtrain[9:10, i:i + 9]
            temp4 = newtrain[12:14, i:i + 9]
            temp = np.r_[temp1, temp2]
            temp = np.r_[temp, temp3]
            temp = np.r_[temp, temp4]
            temp = np.r_[temp, np.power(temp3, 2)]
            # temp = np.r_[temp, np.power(temp3, 3)]
            temp = np.r_[temp, np.power(temp5, 2)]
            # temp = newtrain[:, i:i + 9]
            temp = temp.flatten()
            y[count2] = newtrain[9, i + 9]
            feature[count2, :] = temp
            count2 += 1

    feature = np.delete(feature, np.arange(5495, 5524), 0)
    feature = np.delete(feature, np.arange(5470, 5492), 0)
    feature = np.delete(feature, np.arange(5310, 5334), 0)
    feature = np.delete(feature, np.arange(5024, 5035), 0)
    feature = np.delete(feature, np.arange(3748, 3767), 0)
    feature = np.delete(feature, np.arange(3084, 3096), 0)
    feature = np.delete(feature, np.arange(3056, 3066), 0)
    feature = np.delete(feature, np.arange(1683, 1698), 0)
    feature = np.delete(feature, np.arange(1439, 1448), 0)
    feature = np.delete(feature, np.arange(1405, 1412), 0)
    feature = np.delete(feature, np.arange(1278, 1289), 0)
    feature = np.delete(feature, np.arange(1203, 1220), 0)
    feature = np.delete(feature, np.arange(457, 467), 0)

    y = np.delete(y, np.arange(5495, 5524), 0)
    y = np.delete(y, np.arange(5470, 5492), 0)
    y = np.delete(y, np.arange(5310, 5334), 0)
    y = np.delete(y, np.arange(5024, 5035), 0)
    y = np.delete(y, np.arange(3748, 3767), 0)
    y = np.delete(y, np.arange(3084, 3096), 0)
    y = np.delete(y, np.arange(3056, 3066), 0)
    y = np.delete(y, np.arange(1683, 1698), 0)
    y = np.delete(y, np.arange(1439, 1448), 0)
    y = np.delete(y, np.arange(1405, 1412), 0)
    y = np.delete(y, np.arange(1278, 1289), 0)
    y = np.delete(y, np.arange(1203, 1220), 0)
    y = np.delete(y, np.arange(457, 467), 0)
    # y = np.delete(y, np.arange((new_col - 9 * 12)-strange_num, new_col - 9 * 12), 0)


    bias = np.ones((feature.shape[0],))
    # feature_power2 = np.power(feature, 2)
    # feature_power3 = np.power(feature, 3)
    # feature = np.c_[feature, feature_power2]
    # feature = np.c_[feature, feature_power3]
    feature = np.c_[bias, feature]  # add bias
    # print(temp.shape)
    feature_row, feature_col = feature.shape
    np.savetxt('feature.csv', feature, delimiter=",", fmt='%s' + ',%s' * (feature_col - 1), comments='')

    print('feature.shape', feature.shape)
    print(y.shape)
    return feature, y

def validation_set(X, Y):
    X_row, X_col = X.shape
    num_val = X_row // 4
    num_train = (X_row * 3) // 4  # num of validation
    print(num_val)
    index_list = np.arange(X_row)
    np.random.seed(1)
    np.random.shuffle(index_list)
    training_index = index_list[:num_train]
    shuffle_index = index_list[num_train:]

    trainingX = X[training_index, :]
    trainingY = Y[training_index]
    trainX_val = X[shuffle_index, :]
    trainY_val = Y[shuffle_index]
    print('trainingX.shape', trainingX.shape)
    print('data set, start training')
    return trainingX, trainingY, trainX_val, trainY_val

'''def drawing(RMSE, _iteration, lr, name):
    # ax1 = plt.figure()
    iter_axis = np.arange(0, _iteration)
    plt.title(str(name)+'using initial lr ='+str(lr))
    plt.xlim(0, _iteration)
    plt.ylim(min(RMSE), max(RMSE))
    plt.xlabel('iteration')
    plt.ylabel('RMSE')
    plt.plot(iter_axis, RMSE)
    plt.show()'''

def training(X, Y, _iteration, lr, _lambda, w_initial):

    X_row, X_col = X.shape
    w = w_initial # (X_col,)
    y_hat = np.zeros((X_row,))  # X_row

    ada_bias = 1e-7
    lr_ada = 0
    RMSE_training = []

    for iter in range(_iteration):

        w_grad = np.zeros((X_col,))
        y_hat = np.dot(X, w)
        y_dif = (Y - y_hat).reshape((X_row, 1))  # X_row
        # print('y_dif.shape', y_dif.shape)
        tmp = np.dot(X.T, (y_dif)).reshape((X_col,))
        # print('tmp.shape', tmp.shape)

        w_grad = w_grad - 2.0 * tmp
        lr_ada = lr_ada + w_grad ** 2

        Ein = (sum((y_dif) ** 2) / X_row) ** 0.5
        Ein = Ein[0]
        RMSE_training.append(Ein)
        w = w * (1-(lr / np.sqrt(lr_ada + ada_bias)) * _lambda) - (lr / np.sqrt(lr_ada+ada_bias)) * w_grad
        print('iter, Ein', iter, Ein)

    # drawing(RMSE_training, _iteration, lr, 'RMSE_training')
    print('y_hat', y_hat)
    return w

def training_val(X, Y, _iteration, lr, val_X, val_Y, _lambda, w_initial):

    X_row, X_col = X.shape

    print('num_train', X_row)
    w = w_initial  # (feature_col,)
    y_hat = np.zeros((X_row,))  # feature_row -> val_split
    # tmp_ada_sum = 0
    ada_bias = 1e-7
    lr_ada = 0

    RMSE_validaiton = []
    min_Eval = np.inf
    record_iter = 0

    for iter in range(_iteration):

        w_grad = np.zeros((X_col,))

        y_hat = np.dot(X, w)
        print('y_hat.shape', y_hat.shape)
        y_dif = (Y - y_hat).reshape((X_row, 1))  # feature_row -> val_split
        tmp = np.dot(X.T, (y_dif)).reshape((X_col,))
        w_grad = w_grad - 2.0 * tmp
        lr_ada = lr_ada + w_grad ** 2

        w = w * (1-(lr / np.sqrt(lr_ada+ada_bias)) * _lambda) - (lr / np.sqrt(lr_ada+ada_bias)) * w_grad

        y_val = np.dot(val_X, w)
        y_val[np.where(y_val < 0)] = 0
        Eval = np.sqrt(sum((val_Y - y_val) ** 2) / X_row)
        RMSE_validaiton.append(Eval)
        if Eval < min_Eval:
            min_Eval = Eval
            record_iter = iter
        print('Eval:', iter, Eval)

    # print('lr', lr_ada)
    print('y_val_predict', y_val)
    print('y_val_label:', val_Y)
    print('min_Eval iter', record_iter, min_Eval)
    print('y_hat', y_hat)
    #drawing(RMSE_validaiton, _iteration, lr, 'RMSE_validaiton')
    return w


# ---- testing part -----
def test(w):
    testing_data = pd.read_csv('test.csv', encoding="ISO-8859-1", header=None)
    testing_data = testing_data.values
    testing_data = testing_data[:, 2:]

    row_test, col_test = testing_data.shape
    print('test_shape', testing_data.shape)
    newtest = []
    bias_test = np.ones((260,))
    testing_data[np.where(testing_data == 'NR')] = 0
    testing_data = np.array(testing_data).astype(float)
    for i in range(0, row_test, 18):
        temp1 = testing_data[i+1:i+4, :]
        temp2 = testing_data[i+5:i+7, :]
        temp3 = testing_data[i+8:i+10, :]
        temp5 = testing_data[i+9:i+10, :]
        temp4 = testing_data[i+12:i+14, :]
        subset = np.r_[temp1, temp2]
        subset = np.r_[subset, temp3]
        subset = np.r_[subset, temp4]
        subset = np.r_[subset, np.power(temp3, 2)]
        # subset = np.r_[subset, np.power(temp3, 3)]
        subset = np.r_[subset, np.power(temp5, 2)]
        # subset = testing_data[i:i + 18, :]
        subset = subset.flatten()
        newtest.append(subset)

    newtest = np.array(newtest).astype(float)

    # newtest_power2 = np.power(newtest, 2)
    # newtest_power3 = np.power(newtest, 3)
    # newtest = np.c_[newtest, newtest_power2]
    # newtest = np.c_[newtest, newtest_power3]
    newtest = np.c_[bias_test, newtest]


    print(newtest.shape)
    # print('w',w)
    y_output = np.dot(newtest, w.T)
    y_output[np.where(y_output < 0)] = 0
    #for i in range(len(y_output)):
    #    if y_output[i] % 1 > 0.65:
    #        y_output[i] = int(y_output[i]+1)
    #    else:
    #        y_output[i] = int(y_output[i])
    print('y_out', y_output)
    label = ['id_' + str(x) for x in range(260)]
    label = np.array(label)
    # print(label.shape)
    output = np.c_[label, y_output]
    return output


X, Y = load_training_data('train.csv')

if val:
    X, Y, val_X, val_Y = validation_set(X, Y)
    w = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), Y)
    w = training_val(X, Y, _iteration, lr, val_X, val_Y, _lambda,w)
else:
    w = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), Y)
    w = training(X, Y, _iteration, lr, _lambda, w)

print('w:', w)
np.save('model.npy', w)
output = test(w)

np.savetxt('submission.csv', output, delimiter=",", fmt='%s'+ ',%s', header='id'+',value', comments='')
