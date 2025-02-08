# -*- coding: utf-8 -*-


from scipy.stats import pearsonr
import numpy as np


# 选择距离中心最近的点
def spilt_ball_7(data, data_index):
    n=len(data)
    # # 计算中心点
    center = data.mean(axis=0)
    ball1 = []
    ball2 = []
    index1 = []
    index2 = []
    relation1 = np.zeros(len(data))
    for j in range(0, len(data)):
        pearson_corr, p_value = pearsonr(center, data[j])
        relation1[j] = 1 - pearson_corr
    c1 = np.argmin(relation1)
    relation2 = np.zeros(len(data))
    for j in range(0, len(data)):
        pearson_corr, p_value = pearsonr(data[c1], data[j])
        relation2[j] = 1 - pearson_corr
    #     找与c1最不相关的点
    # c2 = np.argmax(relation2)
    # c2_value=np.max(relation2)

    c2 = np.argmax(relation1)
    c2_value = np.max(relation1)
    if c2_value<1:
        return [ball1, ball2, index1, index2]

    for j in range(0, len(data)):
       if j!=c2:
           pearson_corr, p_value = pearsonr(data[c2], data[j])
           relation=1 - pearson_corr
           if (relation2[j] < relation):
               ball1.extend([data[j, :]])
               index1.append(data_index[j])
           else:
               ball2.extend([data[j, :]])
               index2.append(data_index[j])
       else:
           ball2.extend([data[j, :]])
           index2.append(data_index[j])

    ball1 = np.array(ball1)
    ball2 = np.array(ball2)

    return [ball1, ball2, index1, index2]



def get_person_volume(gb):
    num = len(gb)
    # 计算gb中所有点的均值
    center = gb.mean(0)
    sum_correlation=0
    for j in range(0,len(gb)):
        pearson_corr, p_value = pearsonr(center, gb[j])
        sum_correlation =sum_correlation+(1 - pearson_corr)
    ps=num/sum_correlation
    return ps


def spilt_ball_2(data,data_index):
    ball1 = []
    ball2 = []
    index1 = []
    index2 = []
    center = data.mean(axis=0)
    p_max1 = np.argmax(((data - center) ** 2).sum(axis=1) ** 0.5)
    p_max2 = np.argmax(((data - data[p_max1]) ** 2).sum(axis=1) ** 0.5)
    c1 = (data[p_max1] + center) / 2
    c2 = (data[p_max2] + center) / 2


    # 遍历球内的每个样本
    for j in range(0, len(data)):
        if np.linalg.norm(data[j] - c1) < np.linalg.norm(data[j] - c2):
            ball1.extend([data[j, :]])
            index1.append(data_index[j])
        else:
            ball2.extend([data[j, :]])
            index2.append(data_index[j])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2,index1,index2]




# 获取球内的密度
def get_density_volume(gb):
    num = len(gb)
    # 计算gb中所有点的均值
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    # 每个点到中心点的距离
    distances = sqDistances ** 0.5
    sum_radius = 0
    if len(distances) == 0:
        print("0")
    # radius = max(distances)
    for i in distances:
        sum_radius = sum_radius + i
    # 平均距离
    mean_radius = sum_radius / num
    dimension = len(gb[0])
    # print('*******dimension********',dimension)
    if mean_radius != 0:
        # density_volume = num/(radius**dimension)
        # density_volume = num/((radius**dimension)*sum_radius)
        density_volume = num / sum_radius
        # density_volume = num/(sum_radius)
    else:
        density_volume = num

    return density_volume

# 无参遍历粒球是否需要分裂，根据子球和父球的比较，不带断裂判断的分裂,1分2

def division_2_3(gb_list,gb_data_index):
    gb_list_new = []
    gb_list_index_new=[]

    for i,gb_data in enumerate (gb_list):
        # 粒球内样本数大于等于8的粒球进行处理
        if len(gb_data) >= 8 :
            ball_1, ball_2,index_1,index_2 = spilt_ball_7(gb_data,gb_data_index[i])
            # 如果划分的两个球中 其中一个球内的样本数小于等于1 则该球不该划分
            if len(ball_1) == 1 or len(ball_2) == 1:
                gb_list_new.append(gb_data)
                gb_list_index_new.append(gb_data_index[i])
                continue

            if len(ball_1) == 0 or len(ball_2) == 0:
                gb_list_new.append(gb_data)
                gb_list_index_new.append(gb_data_index[i])
                continue

            parent_ps = get_person_volume(gb_data[:, :])
            child_1_ps = get_person_volume(ball_1[:, :])
            child_2_ps = get_person_volume(ball_2[:, :])
            w1 = len(ball_1) / (len(ball_1) + len(ball_2))
            w2 = len(ball_2) / (len(ball_1) + len(ball_2))

            w_child_ps = (w1 * child_1_ps + w2 * child_2_ps)  # 加权子粒球DM

            t2 = (w_child_ps > parent_ps)  # 加权上升

            if t2:
                gb_list_new.extend([ball_1, ball_2])
                gb_list_index_new.extend([index_1, index_2])
            else:
                gb_list_new.append(gb_data)
                gb_list_index_new.append(gb_data_index[i])

        else:
            gb_list_new.append(gb_data)
            gb_list_index_new.append(gb_data_index[i])



    return gb_list_new,gb_list_index_new

def division_2_2(gb_list,gb_data_index):
    gb_list_new = []
    gb_list_index_new=[]

    for i,gb_data in enumerate (gb_list):


        # 粒球内样本数大于等于8的粒球进行处理
        if len(gb_data) >= 8 :
            ball_1, ball_2,index_1,index_2 = spilt_ball_2(gb_data,gb_data_index[i])  # 无模糊

            # 如果划分的两个球中 其中一个球的内的样本数小于等于1 则该球不该划分
            if len(ball_1) == 1 or len(ball_2) == 1:
                gb_list_new.append(gb_data)
                gb_list_index_new.append(gb_data_index[i])
                continue
            if len(ball_1) == 0 or len(ball_2) == 0:
                gb_list_new.append(gb_data)
                gb_list_index_new.append(gb_data_index[i])
                continue

            parent_dm = get_density_volume(gb_data[:, :])
            child_1_dm = get_density_volume(ball_1[:, :])
            child_2_dm = get_density_volume(ball_2[:, :])
            w1 = len(ball_1) / (len(ball_1) + len(ball_2))
            w2 = len(ball_2) / (len(ball_1) + len(ball_2))

            w_child_dm = (w1 * child_1_dm + w2 * child_2_dm)  # 加权子粒球DM

            t1 = ((child_1_dm > parent_dm) & (child_2_dm > parent_dm))
            t2 = (w_child_dm > parent_dm)  # 加权DM上升
            t3 = ((len(ball_1) > 0) & (len(ball_2) > 0))  # 球中数据个数低于4个的情况不能分裂
            if t2:
                gb_list_new.extend([ball_1, ball_2])
                gb_list_index_new.extend([index_1,index_2])
            else:

                gb_list_new.append(gb_data)
                gb_list_index_new.append(gb_data_index[i])

        else:
            gb_list_new.append(gb_data)
            gb_list_index_new.append(gb_data_index[i])


    return gb_list_new,gb_list_index_new






def get_radius(gb_data):
    # 通过计算每个样本点与中心点之间的距离，并取最大值作为半径。
    # origin get_radius 7*O(n)
    sample_num = len(gb_data)
    center = gb_data.mean(0)
    diffMat = np.tile(center, (sample_num, 1)) - gb_data
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    radius = max(distances)

    # (zt)new get_radius *O(nd)
    # center = gb_data.mean(0)
    # radius = 0
    # for data in gb_data:
    #     temp = 0
    #     index = 0
    #     while index != len(data):
    #         temp += (data[index] - center[index]) ** 2
    #         index += 1
    #     radius_temp = temp ** 0.5
    #     if radius_temp > radius:
    #         radius = radius_temp
    return radius


# 缩小粒球
def minimum_ball(gb_list, radius_detect,index):
    gb_list_temp = []
    gb_list_temp_index=[]
    for i,gb_data in enumerate(gb_list):
        # if len(hb) < 2: stream
        if len(gb_data) <= 2:
            # gb_lis t_temp.append(gb_data)

            if (len(gb_data) == 2) and (get_radius(gb_data) > 2* radius_detect):
                # print(get_radius(gb_data))
                gb_list_temp.append(np.array([gb_data[0], ]))
                gb_list_temp.append(np.array([gb_data[1], ]))
                gb_list_temp_index.append(index[i][0])
                gb_list_temp_index.append(index[i][1])
            else:
                gb_list_temp.append(gb_data)
                gb_list_temp_index.append(index[i])
        else:
            # if get_radius(gb_data) <= radius_detect:
            if get_radius(gb_data) <= 2 * radius_detect:
                gb_list_temp.append(gb_data)
                gb_list_temp_index.append(index[i])
            else:
                ball_1, ball_2,index_1,index_2 = spilt_ball_2(gb_data,index[i])  # 无模糊
                # ball_1, ball_2 = spilt_ball_fuzzy(gb_data)  # 有模糊
                if len(ball_1) == 1 or len(ball_2) == 1:
                    if get_radius(gb_data) > radius_detect:
                        gb_list_temp.extend([ball_1, ball_2])
                        gb_list_temp_index.extend([index_1,index_2])
                    else:
                        gb_list_temp.append(gb_data)
                        gb_list_temp_index.append(index)
                else:
                    gb_list_temp.extend([ball_1, ball_2])
                    gb_list_temp_index.extend([index_1, index_2])

    return gb_list_temp,gb_list_temp_index

