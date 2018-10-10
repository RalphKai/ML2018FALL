import numpy as np
import pandas as pd
import sys

def test(w):
    testing_data = pd.read_csv(sys.argv[1], header=None)
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


w = [-4.66165404e+00, 1.16639701e+00, 8.69272716e-01, -2.08312643e-01
, 1.10041859e+00, 8.51936870e-01, -1.89986913e-01, -2.10305004e+00
, 1.82720959e-01, 1.80001946e+00, 4.87897199e-01,-6.27287111e-01
,-1.88766807e+00,-1.72585456e+00,-1.47645042e+00, 3.74845335e+00
, 1.80224661e+00,-1.28646247e+00, 4.47934378e+00, 3.54854595e-01
,-1.87696886e-01,-1.53016648e+00, 2.91955791e+00, 1.39513925e-01
, 2.26085255e+00,-3.39423837e+00,-2.74501779e+00, 3.38686780e-01
,-4.23665165e-02, 2.57614314e-02,-9.17900779e-02, 1.42719876e-01
,-6.30822630e-02,-5.63606467e-02,-3.92370000e-02, 1.03474312e-01
, 9.16123674e-03, 5.42681830e-03,-1.13736661e-02, 9.92461059e-02
,-6.93159119e-02, 9.01689756e-02,-4.18701052e-02,-2.47205706e-02
,-1.93190710e-02,-2.46769366e-02,-2.65038343e-04,-3.56886148e-03
,-4.08083979e-02, 4.55128273e-02,-4.39733275e-02,-7.68163007e-02
, 1.12088923e-01, 9.78788628e-02,-5.02921640e-02, 1.58488197e-02
, 3.70644202e-02, 1.36920075e-02,-2.39608707e-02, 8.86454568e-02
, 1.31475577e-01,-1.86783580e-01, 1.55527698e-01, 5.23180306e-01
, 1.80602827e-01, 3.13218261e-03,-1.45311639e-01,-7.82105784e-04
, 2.01470923e-01,-2.58825962e-01, 1.54695524e-02,-1.09642483e-01
, 5.56662566e-01,-7.94502122e-01,-2.81604816e-02, 4.50148585e-01
,-1.69618618e+00,-1.36390799e+00,-1.58095642e+00, 3.15374636e+00
, 2.10461730e+00,-6.09447103e-01,-3.51156933e-05, 1.35954691e-05
, 1.24589960e-04,-2.39565182e-04, 2.57079264e-04, 2.89310646e-04
,-6.60914586e-04,-7.61670499e-04, 1.07843647e-03,-1.11059218e-05
,-1.22084446e-04, 2.87286082e-04,-4.59627189e-04,-1.41219333e-04
, 6.90474258e-04,-8.39417193e-04,-7.87829966e-04, 1.85594486e-03
,-1.11059218e-05,-1.22084446e-04, 2.87286082e-04,-4.59627189e-04
,-1.37489108e-04, 6.90391158e-04,-8.39434297e-04,-7.87845270e-04
, 1.85714812e-03]

w = np.array(w)
output = test(w)

np.savetxt(sys.argv[2], output, delimiter=",", fmt='%s'+ ',%s', header='id'+',value', comments='')
