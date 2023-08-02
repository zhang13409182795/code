
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from scipy.io import loadmat
from scipy.io import savemat
from os import path
import numpy as np

def genMeasurements(x,A,SNR_all):

    [N,K] = np.shape(x)
    [M,N] = np.shape(A)
    batch = len(SNR_all)
    batchsize = int(K/batch)
    y = np.zeros(dtype=np.complex128, shape=(M, K))
    for k in range(int(batch)):
        xk = x[..., k*batchsize:(k+1)*batchsize]
        SNR_dB = SNR_all[k]
        print(SNR_dB)
        SNR = np.power(10, SNR_dB / 10.)
        sigma = 1 / SNR
        cNoise = np.sqrt(sigma) * ((np.random.normal(size=(N, batchsize)) + 1j * np.random.normal(size=(N, batchsize))) / np.sqrt(2))
        yk=np.matmul(A, xk+cNoise)
        y[..., k*batchsize:(k+1)*batchsize] = yk

    return y


if __name__ == '__main__':
    M = 128
    N = 256
    type = 'UPA'  # 'UPA'

    D = loadmat('SV_' + type + '_training_channel.mat')
    training_x = D['x']
    D = loadmat('SV_' + type + '_validation_channel.mat')
    validation_x = D['x']
    D = loadmat('SV_' + type + '_test_channel.mat')
    test_x = D['x']

    training_size = np.shape(training_x)
    validation_size = np.shape(validation_x)
    test_size = np.shape(test_x)

    D = loadmat('CSmatrix' + str(N) + str(M) + '.mat')
    A = D['A']

    training_y = np.zeros(dtype=np.complex128, shape=(M,training_size[1]))
    validation_y = np.zeros(dtype=np.complex128, shape=(M, validation_size[1]))
    test_y = np.zeros(dtype=np.complex128, shape=(M, test_size[1]))


    # n1 = 10
    # n2 = 20
    # SNR_dB = np.random.uniform(n1, n2, size=(10))
    # training_y = genMeasurements(training_x, A, SNR_dB)
    # D = dict(y=training_y)
    # savemat('SV_'+type+'_training_'+str(M)+'_measurements_'+str(n1)+'to'+str(n2)+'dB.mat', D)
    #
    # n = [10]
    # validation_y = genMeasurements(validation_x, A, n)
    # D = dict(y=validation_y)
    # savemat('SV_'+type+'_val_'+str(M)+'_measurements_'+str(n[0])+'dB.mat', D)

    # n = [-5]
    # validation_y = genMeasurements(training_x, A, n)
    # D = dict(y=training_y)
    # savemat('SV_'+type+'_training_'+str(M)+'_measurements_'+str(n[0])+'dB.mat', D)

    n = [-10]
    test_y = genMeasurements(test_x, A, n)
    D = dict(y=test_y)
    savemat('SV_'+type+'_test_'+str(M)+'_measurements_'+str(n[0])+'dB.mat', D)




