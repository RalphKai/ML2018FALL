import numpy as np
import pandas as pd
import sys

def load_data(filename):
    data = pd.read_csv(filename)
    data = data.values
    return data

class Logistic_Regression():
    def __init__(self):
        pass

    def parameter_init(self, dim, b):
        self.b = b
        self.W = np.zeros((dim, 1))
        # self.W = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), Y)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-1 * z) + 1e-8)

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
        # add = X[:, 1:4]
        amount = self.feature_scaling(X[:, 0])
        new_X = np.c_[amount, add]
        new_X = np.c_[new_X, X[:, 5]]
        # X[X[:, 5] < 1, 5] = 0
        temp = X[:, 5]+2
        feature_power = np.power(temp, 2)
        new_X = np.c_[new_X, feature_power]
        new_X = np.c_[new_X, feature_power]
        feature_power3 = np.power(temp, 3)
        new_X = np.c_[new_X, feature_power3]
        new_X = np.c_[new_X, feature_power3]
        new_X = np.c_[new_X, feature_power3]

        return new_X

    def feature_scaling(self, X):

        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        # return (X - mean) / std
        return (X - self.min) / (self.max - self.min)



    def RMSELoss(self, X, Y):
        return np.sqrt(np.mean((Y - self.predict(X)) ** 2))

    def predict(self, X):
        return self.sigmoid(np.dot(X, self.W) + self.b)

    def predict_output(self, X, W, b):
        z = np.dot(X, W) + b
        temp = 1 / (1 + np.exp(-1 * z))
        print(temp)
        for i in range(len(temp)):
            if temp[i] > 0.38:
                temp[i] = 1
            else:
                temp[i] = 0

        return temp

    def train(self, X, Y, valX, valY, epochs=156, lr=0.01, b_initial=-1, _lambda=0.005):
        batch_size = X.shape[0]
        W_dim = X.shape[1]
        self.parameter_init(W_dim, b_initial)
        # X = self.feature_scaling(X, train=True)
        print(batch_size)
        lr_b = 0
        lr_W = np.zeros((W_dim, 1))
        minEval = np.inf
        m_w = np.zeros((W_dim, 1))
        m_b = 0
        v_w = np.zeros((W_dim, 1))
        v_b = 0
        alpha = 0.03
        beta_1 = 0.9
        beta_2 = 0.999  # initialize the values of the parameters
        epsilon = 1e-8
        t = 0
        for epoch in range(epochs):
            # mse loss
            grad_b = -np.sum(Y - self.predict(X))   # / batch_size
            grad_W = -np.dot(X.T, (Y - self.predict(X)))    # / batch_size

            # # adagrad
            # lr_b += grad_b ** 2
            # lr_W += grad_W ** 2
            #
            # # update
            # self.b = self.b - lr / np.sqrt(lr_b) * grad_b
            # self.W = self.W - lr / np.sqrt(lr_W) * grad_W

            # adam
            t = t + 1
            m_w = beta_1 * m_w + (1 - beta_1) * grad_W
            m_b = beta_1 * m_b + (1 - beta_1) * grad_b
            v_w = beta_2 * v_w + (1 - beta_2) * grad_W ** 2
            v_b = beta_2 * v_b + (1 - beta_2) * grad_b ** 2
            m_w_hat = m_w / (1 - beta_1 ** t)
            m_b_hat = m_b / (1 - beta_1 ** t)
            v_w_hat = v_w / (1 - beta_2 ** t)
            v_b_hat = v_b / (1 - beta_2 ** t)

            # update
            w_record = self.W
            b_record = self.b
            self.b = self.b - alpha * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
            self.W = self.W - alpha * m_w_hat / (np.sqrt(v_w_hat) + epsilon) #- _lambda * self.W


test_X = load_data(sys.argv[3])

model = Logistic_Regression()
# model2 = Generative_Model()
test_X = model.data_prepocess(test_X)


model = Logistic_Regression()
# X = model.data_prepocess(X)
# X = model.feature_scaling(X)
valX = None
valY = None
#
# print(X.shape)
W = np.load('model.npy')
b = -1.4528435549806331
# model.train(X, Y, valX, valY, 120)


# model.validation(valX, valY)
# RMSELoss_Ein = model.RMSELoss(X,Y)


# test_X = test_X[:, 5].reshape((test_X.shape[0], 1))
# test_X = model.feature_scaling(test_X)
y_predict = model.predict_output(test_X, W, b).astype(int)
# y_predict = model2.test(test_X).astype(int)
print(y_predict)

label = ['id_' + str(x) for x in range(len(y_predict))]
label = np.array(label)
output = np.c_[label, y_predict]

np.savetxt(sys.argv[4], output, delimiter=",", fmt='%s'+ ',%s', header='id'+',Value', comments='')
# print('RMSE_in:', RMSELoss_Ein)