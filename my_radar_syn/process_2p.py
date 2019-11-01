from __future__ import division, print_function
from pymoduleconnector import ModuleConnector
import winsound
import numpy as np
import time, os
import sympy
from sympy.abc import x, y
import cv2
import matplotlib.pyplot as plt


def my_mti(arr):
    fast_t = arr.shape[0]
    low_t = arr.shape[1]
    arr1 = np.zeros(shape=(fast_t, low_t))
    for i in range(fast_t):
        for j in range(low_t - 1):
            arr1[i, j] = arr[i, j+1] - arr[i, j]
    arr1[:, low_t-1] = arr1[:, low_t - 2]
    return arr1


def mti_third(arr):
    arr1 = arr.copy()
    low_t = arr.shape[1]
    for j in range(3, low_t):
        arr1[:, j] = arr[:, j] - 3*arr[:, j-1] + 3*arr[:, j-2] - arr[:, j-3]
    return arr1[:, 3:]


def get_main(filename):
    data_in = np.load(filename)
    data_in = abs(data_in)
    data_in = data_in.T
    data_mti = my_mti(data_in)
    #data_mti_third = mti_third(data_in)
    return data_in, data_mti


def find2_max(arr, n_left, n_right, T):
    fast_t = arr.shape[0]
    low_t = arr.shape[1]
    toa = [[] for _ in range(low_t)]
    dn = arr.copy()
    pic = np.zeros(shape=(fast_t, low_t))
    for j in range(low_t):
        an = np.max(abs(dn[:, j]))
        while an >= T:
            max_index1 = np.argmax(abs(dn[:, j]))
            if an >= np.max(abs(arr[max_index1 - n_left:max_index1 + n_right, j])):
                toa[j].append(max_index1)
                pic[max_index1, j] = 1
            dn[max_index1 - n_left:max_index1 + n_right, j] = 0
            an = np.max(abs(dn[:, j]))
    return toa, pic


def find_2max(arr, n_left, n_right, T):
    fast_t = arr.shape[0]
    low_t = arr.shape[1]
    max2_arr = np.zeros(shape=(fast_t, low_t))
    max2_range = np.zeros(shape=(2, low_t))
    arr1 = arr.copy()
    for i in range(fast_t):
        arr1[i, :] = ((i*0.0514)**1.7)*arr1[i, :]
    for j in range(low_t):
        arr1[0:10, j] = 0
        max_index1 = np.argmax(abs(arr1[:, j]))
        if arr1[max_index1, j] >= T:
            max2_arr[max_index1, j] = 1
        arr1[max_index1 - n_left:max_index1 + n_right, j] = 0
        max_index2 = np.argmax(abs(arr1[:, j]))
        if arr1[max_index2, j] >= T:
            max2_arr[max_index2, j] = 1
    return max2_arr


def find_2max_ori(arr, n_left, n_right):
    fast_t = arr.shape[0]
    low_t = arr.shape[1]
    max2_arr = np.zeros(shape=(fast_t, low_t))
    max2_range = np.zeros(shape=(2, low_t))
    arr1 = arr.copy()
    for i in range(fast_t):
        arr1[i, :] = ((i*0.0514)**1.7)*arr1[i, :]
    for j in range(low_t):
        arr1[0:10, j] = 0
        max_index1 = np.argmax(abs(arr1[:, j]))
        max2_arr[max_index1, j] = 1
        #max2_range[0, j] = max_index1*0.0514
        arr1[max_index1 - n_left:max_index1 + n_right, j] = 0
        max_index2 = np.argmax(abs(arr1[:, j]))
        max2_arr[max_index2, j] = 1
        #max2_range[1, j] = max_index2*0.0514
    max2_arr_fil = my_wind(max2_arr, 4, 2)
    for j in range(low_t):
        count = 0
        for i in range(fast_t):
            if max2_arr_fil[i, j] == 1:
                max2_range[count, j] = i * 0.0514
                count += 1

    """
    for j in range(low_t - 1):
        if abs(max2_range[0, j] - near_num(max2_range[0, j + 1], max2_range[1, j + 1], dis_range[0, j])) > 0.5\
                and abs(max2_range[1, j] - near_num(max2_range[0, j + 1], max2_range[1, j + 1], dis_range[1, j])) > 0.5:
            dis_range[0, j] = near_num(max2_range[0, j + 1], max2_range[1, j + 1], dis_range[0, j])
            dis_range[1, j] = max2_range[1, j + 1] if dis_range[0, j] == max2_range[0, j + 1] else max2_range[0, j + 1]
        elif abs(max2_range[0, j] - near_num(max2_range[0, j + 1], max2_range[1, j + 1], dis_range[0, j])) > 0.5\
                and abs(max2_range[1, j] - near_num(max2_range[0, j + 1], max2_range[1, j + 1], dis_range[1, j])) < 0.5:
            dis_range[0, j] = far_num(max2_range[0, j + 1], max2_range[1, j + 1], dis_range[1, j])
        elif abs(max2_range[0, j] - near_num(max2_range[0, j + 1], max2_range[1, j + 1], dis_range[0, j])) < 0.5\
                and abs(max2_range[1, j] - near_num(max2_range[0, j + 1], max2_range[1, j + 1], dis_range[1, j])) > 0.5:
            dis_range[1, j] = far_num(max2_range[0, j + 1], max2_range[1, j + 1], dis_range[0, j])
    """

    return max2_arr, max2_range, max2_arr_fil


def near_num(num1, num2, num):
    return num1 if abs(num - num1) < abs(num - num2) else num2


def far_num(num1, num2, num):
    return num1 if abs(num - num1) > abs(num - num2) else num2


def my_wind(arr, half_len, thresh):
    arr1 = arr.copy()
    fast_t = arr1.shape[0]
    low_t = arr1.shape[1]
    for i in range(half_len, fast_t - half_len):
        for j in range(half_len, low_t - half_len):
            if arr1[i, j] == 1:
                sum_win = np.sum(arr1[i - half_len:i + half_len + 1, j - half_len:j + half_len + 1])
                if sum_win <= thresh:
                    arr1[i, j] = 0
    return arr1


def get_2p_point(arr1, arr2):
    slow_t = arr1.shape[1]
    x1_list = list()
    x2_list = list()
    y1_list = list()
    y2_list = list()
    x1_list_ghost = list()
    x2_list_ghost = list()
    y1_list_ghost = list()
    y2_list_ghost = list()
    for j in range(slow_t):
        if arr1[0, j] != 0 and arr1[1, j] != 0 and arr2[0, j] != 0 and arr2[1, j] != 0:
            aa = sympy.solve([(x + 0.8) ** 2 + y ** 2 - arr1[0, j] ** 2, (x - 0.8) ** 2 + y ** 2 - arr2[0, j] ** 2], [x, y])
            result = [round(aa[0][0], 2), round(abs(aa[0][1]), 2)]
            x1_list.append(result[0])
            y1_list.append(result[1])
            aa = sympy.solve([(x + 0.8) ** 2 + y ** 2 - arr1[1, j] ** 2, (x - 0.8) ** 2 + y ** 2 - arr2[1, j] ** 2],
                             [x, y])
            result = [round(aa[0][0], 2), round(abs(aa[0][1]), 2)]

            x2_list.append(result[0])
            y2_list.append(result[1])
            aa = sympy.solve([(x + 0.8) ** 2 + y ** 2 - arr1[0, j] ** 2, (x - 0.8) ** 2 + y ** 2 - arr2[1, j] ** 2],
                             [x, y])
            result = [round(aa[0][0], 2), round(abs(aa[0][1]), 2)]
            x1_list_ghost.append(result[0])
            y1_list_ghost.append(result[1])
            aa = sympy.solve([(x + 0.8) ** 2 + y ** 2 - arr1[1, j] ** 2, (x - 0.8) ** 2 + y ** 2 - arr2[0, j] ** 2],
                             [x, y])
            result = [round(aa[0][0], 2), round(abs(aa[0][1]), 2)]
            x2_list_ghost.append(result[0])
            y2_list_ghost.append(result[1])
    return x1_list, x2_list, y1_list, y2_list, x1_list_ghost, x2_list_ghost, y1_list_ghost, y2_list_ghost


trail_41_ori, trail_41_mti = get_main('outside_p2_walk_radar_1_1571984691.5356767.npy')
trail_42_ori, trail_42_mti = get_main('outside_p2_walk_radar_2_1571984691.5356767.npy')
pic_41, dis_1, pic_41_re = find_2max_ori(trail_41_mti, 10, 10)
#toa_41_r, pic_41_r = find2_max(trail_41_mti3, 10, 25, 2e-3)
#pic_42 = find_2max_ori(trail_42_mti, 10, 10)
pic_42, dis_2, pic_42_re = find_2max_ori(trail_42_mti, 10, 10)

print('This is from radar1')
print(dis_1)
print('This is from radar2')
print(dis_2)

x1_list, x2_list, y1_list, y2_list, x1_list_ghost, x2_list_ghost, y1_list_ghost, y2_list_ghost = get_2p_point(dis_1, dis_2)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(1)
plt.imshow(abs(trail_41_ori), origin='lower')
plt.title('雷达1：原始')
plt.figure(2)
plt.imshow(abs(trail_41_mti), origin='lower')
plt.title('雷达1：mti')
plt.figure(3)
plt.imshow(abs(pic_41), origin='lower')
plt.title('雷达1：求大值(距离衰减)')
plt.figure(4)
plt.imshow(abs(pic_41_re), origin='lower')
plt.title('雷达1：求大值(距离补偿) + 窗')
plt.figure(5)
plt.imshow(abs(trail_42_ori), origin='lower')
plt.title('雷达2：原始')
plt.figure(6)
plt.imshow(abs(trail_42_mti), origin='lower')
plt.title('雷达2：mti')
plt.figure(7)
plt.imshow(abs(pic_42), origin='lower')
plt.title('雷达2：最大值(距离衰减)')
plt.figure(8)
plt.imshow(abs(pic_42_re), origin='lower')
plt.title('雷达2：求大值(距离补偿) + 窗')
plt.figure(9)

plt.axis([-3, 3, 0, 6])
ax1 = plt.gca()
ax1.xaxis.set_ticks_position('top')
ax1.yaxis.set_ticks_position('left')
ax1.invert_yaxis()
ax1.spines['left'].set_position(('data', 0))
plt.plot(x1_list, y1_list, 'o', markersize=7, color='blue', alpha=0.5, label='Target1')
plt.plot(x2_list, y2_list, 'o', markersize=7, color='red', alpha=0.5, label='Target2')
plt.legend()
plt.figure(10)
plt.axis([-3, 3, 0, 6])
ax2 = plt.gca()
ax2.xaxis.set_ticks_position('top')
ax2.yaxis.set_ticks_position('left')
ax2.invert_yaxis()
ax2.spines['left'].set_position(('data', 0))
plt.plot(x1_list_ghost, y1_list_ghost, 'o', markersize=7, color='green', alpha=0.5, label='Target1_g')
plt.plot(x2_list_ghost, y2_list_ghost, 'o', markersize=7, color='yellow', alpha=0.5, label='Target2_g')
plt.legend()
plt.show()
