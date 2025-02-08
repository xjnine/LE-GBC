
from gbutils.granular_ball import GranularBall
from gbutils.GBUtil import *


def splitGBs(data,all_indices):

    gb_list_temp = [data]  # 粒球集合[ [[data1],[data2],...], [[data1],[data2],...],... ],初始只有一个粒球

    # 当粒球不再分裂停止

    index=[all_indices]
    # 基于person划分
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp, index = division_2_3(gb_list_temp, index)  # 使用 division_2_3 进行分裂
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break  # 如果粒球数量不再变化，停止循环

    # 基于欧式聚类划分
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp, index = division_2_2(gb_list_temp, index)
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    radius = []  # 汇总所有粒球半径
    for gb_data in gb_list_temp:
        if len(gb_data) >= 2:  # 粒球中样本多于2个时，认定为合法粒球，并收集其粒球半径
            radius.append(get_radius(gb_data[:, :]))
    radius_median = np.median(radius)  #半径的中位数
    radius_mean = np.mean(radius)  #半径的均值

    radius_detect = min(radius_median, radius_mean)
    # 缩小粒球
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp,index = minimum_ball(gb_list_temp, radius_detect,index)  # 缩小粒球 将粒球半径大于radius_detect的进行缩小
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    gb_list = []
    for obj in gb_list_temp:
        gb = GranularBall(obj,index)
        gb_list.append(gb)
    return gb_list,index


