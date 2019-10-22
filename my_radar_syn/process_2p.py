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
    data_mti_third = mti_third(data_in)
    return data_in, data_mti, data_mti_third


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


trail_41_ori, trail_41_mti, trail_41_mti3 = get_main('D:\Working\PycharmSpace\my_radar_syn\\2p_1564400587.1684678.npy')
#trail_42_ori, trail_42_mti, trail_42_clutter_reduce = get_main('D:\Working\PycharmSpace\my_radar_syn\p2_walk_radar_2_1570795371.564845.npy')
toa_41, pic_41 = find2_max(trail_41_mti, 10, 25, 5e-4)
toa_41_r, pic_41_r = find2_max(trail_41_mti3, 10, 25, 2e-3)
#toa_42, pic_42 = find2_max(trail_42_mti, 10, 25, 5e-4)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
"""
plt.figure(1)
plt.imshow(abs(trail_41_ori), origin='lower')
plt.title('雷达1：原始')
#plt.show()
plt.figure(2)
plt.imshow(abs(trail_42_ori), origin='lower')
plt.title('雷达2：原始')
plt.figure(3)
plt.imshow(abs(trail_41_mti), origin='lower')
plt.title('雷达1：MTI滤波')
#plt.show()
plt.figure(4)
plt.imshow(abs(trail_42_mti), origin='lower')
plt.title('雷达2：MTI滤波')
#plt.show()
plt.figure(5)
plt.imshow(abs(pic_41), origin='lower')
plt.title('雷达1：轨迹')
#plt.show()
plt.figure(6)
plt.imshow(abs(pic_42), origin='lower')
plt.title('雷达2：轨迹')
plt.show()
"""
plt.figure(1)
plt.imshow(abs(trail_41_ori), origin='lower')
plt.title('雷达1：原始')
plt.figure(2)
plt.imshow(abs(trail_41_mti), origin='lower')
plt.title('雷达1：mti')
plt.figure(3)
plt.imshow(abs(trail_41_mti3), origin='lower')
plt.title('雷达1：mti3阶')

plt.figure(4)
plt.imshow(abs(pic_41), origin='lower')
plt.title('雷达1：轨迹')

plt.figure(5)
plt.imshow(abs(pic_41_r), origin='lower')
plt.title('雷达1：轨迹_新')


plt.show()
