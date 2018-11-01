import numpy as np
import pandas as pd
import sys

validation = False


def load_data(filename):
    data = pd.read_csv(filename)
    data = data.values
    return data



def validation_set(X, Y):
    num_data, dim_data = X.shape
    num_train = (num_data * 3) // 4
    index = np.arange(num_data)
    np.random.seed(1)
    np.random.shuffle(index)

    training_index = index[:num_train]
    val_index = index[num_train:]

    trainingX = X[training_index, :]
    trainingY = Y[training_index]
    trainX_val = X[val_index, :]
    trainY_val = Y[val_index]

    return trainingX, trainingY, trainX_val, trainY_val

class Logistic_Regression():
    def __init__(self):
        pass

    def parameter_init(self, dim, b):
        self.b = b
        self.W = np.zeros((dim, 1))
        # self.W = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), Y)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-1 * z) + 1e-8)




class Generative_Model():
    def __init__(self):
        pass

    def parameter_init(self, dim):
        self.b = -1
        self.W = np.zeros((dim, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-1 * z) + 1e-10)

    def data_prepocess(self, X):
        num_data, dim_data = X.shape
        male = np.zeros((num_data,))
        female = np.zeros((num_data,))
        e0 = np.zeros((num_data,))
        e1 = np.zeros((num_data,))
        e2 = np.zeros((num_data,))
        e3 = np.zeros((num_data,))
        e4 = np.zeros((num_data,))
        e5 = np.zeros((num_data,))
        e6 = np.zeros((num_data,))
        m0 = np.zeros((num_data,))
        m1 = np.zeros((num_data,))
        m2 = np.zeros((num_data,))
        m3 = np.zeros((num_data,))
        bill_down = np.zeros((num_data,))

        for n in range(num_data):
            if X[n, 1] == 1:
                male[n] = 1
            if X[n, 1] == 2:
                female[n] = 1
            if X[n, 2] == 0:
                e0[n] = 1
            if X[n, 2] == 1:
                e1[n] = 1
            if X[n, 2] == 2:
                e2[n] = 1
            if X[n, 2] == 3:
                e3[n] = 1
            if X[n, 2] == 4:
                e4[n] = 1
            if X[n, 2] == 5:
                e5[n] = 1  # refer to thesis PDF，4 is other
            if X[n, 2] == 6:
                e6[n] = 1
            # if X[n, 3] == 0:
            #     m0[n] = 1
            if X[n, 3] == 1:
                m1[n] = 1
            if X[n, 3] == 2:
                m2[n] = 1
            if X[n, 3] == 3 or X[n, 3] == 0:
                m3[n] = 1
            if X[n, 11] - X[n, 16] > 0:
                bill_down[n] = -1
            else:
                bill_down[n] = 1
        add = np.array([e0, e1, e2, e3, e4, e5, e6]).T  # num = 9
        # add2 = np.array([p1, p2, p3, p4, p5, p6, p7, p8]).T
        new_X = np.c_[self.feature_scaling(X[:, 0]), add]
        new_X = np.c_[new_X, X[:, 5]]
        temp = X[:, 5]+2
        feature_power = np.power(temp, 2)
        new_X = np.c_[new_X, feature_power]
        feature_power3 = np.power(temp, 4)
        new_X = np.c_[new_X, feature_power3]

        return new_X

    def feature_scaling(self, X):

        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        # return (X - mean) / std
        return (X - self.min) / (self.max - self.min)

    def grouping(self, X, Y):
        X_approved = []
        X_disapproved = []
        for n in range(X.shape[0]):
            if Y[n] == 1:
                X_approved.append(X[n, :])
            else:
                X_disapproved.append(X[n, :])
        return np.array(X_approved), np.array(X_disapproved)

    def test(self, testX):
        z = np.dot(testX, self.W) + self.b
        y = self.sigmoid(z)
        for n in range(y.shape[0]):
            if y[n] > 0.38:
                y[n] = 1
            else:
                y[n] = 0
        return y

    def training_testing(self, X, Y):
        X_approved, X_disapproved = self.grouping(X, Y)
        self.parameter_init(X_approved.shape[0])
        m1 = np.mean(X_approved, axis=0)
        m2 = np.mean(X_disapproved, axis=0)
        prior = X_approved.shape[0] / X.shape[0]
        cov1 = 0
        cov2 = 0
        for n in range(X_approved.shape[0]):
            tmp1 = (X_approved[n, :] - m1).reshape(1, X_approved.shape[1])
            cov1 += np.dot(tmp1.T, tmp1)
        for n in range(X_disapproved.shape[0]):
            tmp2 = (X_disapproved[n, :] - m2).reshape(1, X_disapproved.shape[1])
            cov2 += np.dot(tmp2.T, tmp2)

        cov1 = cov1 / X_approved.shape[0]
        cov2 = cov2 / X_disapproved.shape[0]
        cov = prior * cov1 + (1 - prior) * cov2
        cov_inv = np.linalg.pinv(cov)

        self.W = np.dot( (m1 - m2).T, cov_inv).reshape((X_approved.shape[1], 1))
        self.b = (1/2) * ( -1 * np.dot(np.dot(m1.T, cov_inv), m1) + np.dot(np.dot(m2.T, cov_inv), m2) ) + np.log(X_approved.shape[0]/X_disapproved.shape[0])



X = load_data(sys.argv[1])
Y = load_data(sys.argv[2])
# X = X[:, 5].reshape((X.shape[0], 1))
test_X = load_data(sys.argv[3])

model = Generative_Model()

X = model.data_prepocess(X)
test_X = model.data_prepocess(test_X)
model.training_testing(X, Y)


y_predict = model.test(test_X).astype(int)
print(y_predict)

label = ['id_' + str(x) for x in range(len(y_predict))]
label = np.array(label)
output = np.c_[label, y_predict]

np.savetxt(sys.argv[4], output, delimiter=",", fmt='%s'+ ',%s', header='id'+',Value', comments='')
# print('RMSE_in:', RMSELoss_Ein)
