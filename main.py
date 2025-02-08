import os
import numpy as np
import pandas as pd
from sympy.stats.sampling.sample_scipy import scipy

from Ours import Ours


def run():
    # 读取mat数据集
    dataPath = './datasets/mat/'
    dataName = 'TDT2'
    data = scipy.io.loadmat(os.path.join(dataPath, f'{dataName}.mat'))
    members = data['X']
    gt = data['Y']
    gt = gt.reshape(-1)

    # 读取csv数据集
    # dataPath = './datasets/'
    # file_path_X = os.path.join(dataPath, 'ALLAML_X_1_0_.csv')
    # file_path_Y = os.path.join(dataPath, 'ALLAML_Y_1_0_.csv')
    # data_X = pd.read_csv(file_path_X, header=None)
    # data_X=data_X.T
    # members = data_X.values
    # data_Y = pd.read_csv(file_path_Y, delim_whitespace=True, header=None, usecols=[0])
    # gt = data_Y.iloc[:, 0].values

    k = len(np.unique(gt))
    indicators=[]
    for i in range(20):
        indicator=Ours(members,k,gt)
        indicators.append(indicator)
    indicators = np.array(indicators)
    n = indicators.shape[0]
    indicator_avg = indicators.sum(axis=0) / n
    print(indicator_avg)




if __name__ == '__main__':
    run()













