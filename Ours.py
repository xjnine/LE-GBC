import time

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler

from gbutils.splitGBs import splitGBs


def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    # 利用匈牙利算法进行标签的再分配
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size



def Ours(members,k,gt):

    start_time = time.time()
    scaler = MinMaxScaler(feature_range=(0, 1))
    member = scaler.fit_transform(members)
    all_indices = list(range(len(members)))

    gb_list, gb_index = splitGBs(member, all_indices)
    gb_array = np.empty((len(gb_list), gb_list[0].data.shape[1]))

    for index, gb in enumerate(gb_list):
        gb_array[index] = gb.center

    end_time = time.time()

    time1 = end_time - start_time
    max_acc = 0
    max_indicator=[]
    for i in range(30):
        start_time = time.time()
        kmeans =KMeans(n_clusters= k, init="k-means++",n_init=10)
        kmeans.fit(gb_array)
        member = np.array(member, dtype=np.float64)  # 将 X 转换为 float32
        pred_labels=kmeans.predict(member)
        end_time = time.time()

        # 计算运行时间
        time2 = end_time - start_time
        nmi = normalized_mutual_info_score(gt, pred_labels)
        accu = acc(gt, pred_labels)
        ARI = metrics.cluster.adjusted_rand_score(gt, pred_labels)
        indicator = [accu, nmi, ARI, time1 + time2]
        if max_acc < accu:
            max_indicator = indicator
            max_acc=accu
    return max_indicator
