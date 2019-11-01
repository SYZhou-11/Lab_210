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


def get_main(filename):
    data_in = np.load(filename)
    data_in = abs(data_in)
    data_in = data_in.T
    data_mti = my_mti(data_in)
    #data_mti_third = mti_third(data_in)
    return data_in, data_mti


def find_2max_ori(arr, n_left, n_right):
    fast_t = arr.shape[0]
    low_t = arr.shape[1]
    max2_arr = np.zeros(shape=(fast_t, low_t))
    max2_range = np.zeros(shape=(2, low_t))
    max2_index = np.zeros(shape=(2, low_t))
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
                max2_index[count, j] = i
                count += 1

    return max2_arr, max2_range, max2_arr_fil, max2_index


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


def get_energy(arr, range_l, index):
    energy = 0
    for i in range(range_l + 1):
        energy += abs(arr[index - i])**2
    for i in range(range_l):
        energy += abs(arr[index + i + 1])**2
    energy = energy/(2*range_l + 1)
    return energy


def my_kurtosis(mti_arr1, mti_arr2, max2_index_r1, max2_index_r2, nk, range_l):
    slow_t = max2_index_r1.shape[1]
    eng_r1t1 = list()
    eng_r1t2 = list()
    eng_r2t1 = list()
    eng_r2t2 = list()
    dis_r1t1 = list()
    dis_r1t2 = list()
    dis_r2t1 = list()
    dis_r2t2 = list()
    for j in range(nk-1, slow_t):
        if max2_index_r1[0, j] != 0 and max2_index_r1[1, j] != 0 and max2_index_r2[0, j] != 0 and max2_index_r2[1, j] != 0:
            for k in range(nk - 1, -1, -1):
                eng_r1t1.append(get_energy(mti_arr1[:, j - k], range_l, max2_index_r1[0, j - k]))
                eng_r1t2.append(get_energy(mti_arr1[:, j - k], range_l, max2_index_r1[1, j - k]))
                eng_r2t1.append(get_energy(mti_arr2[:, j - k], range_l, max2_index_r2[0, j - k]))
                eng_r2t2.append(get_energy(mti_arr2[:, j - k], range_l, max2_index_r2[1, j - k]))
            kp_up_r1t1 = 0
            kp_up_r1t2 = 0
            kp_up_r2t1 = 0
            kp_up_r2t2 = 0
            for k in range(nk):
                kp_up_r1t1 += (eng_r1t1[k] - sum(eng_r1t1) / nk) ** 4
                kp_up_r1t2 += (eng_r1t1[k] - sum(eng_r1t2) / nk) ** 4
                kp_up_r2t1 += (eng_r1t1[k] - sum(eng_r2t1) / nk) ** 4
                kp_up_r2t2 += (eng_r1t1[k] - sum(eng_r2t2) / nk) ** 4
            kp_up_r1t1 /= nk
            kp_up_r1t2 /= nk
            kp_up_r2t1 /= nk
            kp_up_r2t2 /= nk
            kp_down_r1t1 = 0
            kp_down_r1t2 = 0
            kp_down_r2t1 = 0
            kp_down_r2t2 = 0
            for k in range(nk):
                kp_down_r1t1 += (eng_r1t1[k] - sum(eng_r1t1) / nk) ** 2
                kp_down_r1t2 += (eng_r1t1[k] - sum(eng_r1t2) / nk) ** 2
                kp_down_r2t1 += (eng_r1t1[k] - sum(eng_r2t1) / nk) ** 2
                kp_down_r2t2 += (eng_r1t1[k] - sum(eng_r2t2) / nk) ** 2
            kp_down_r1t1 = kp_down_r1t1 ** 2 / (nk ** 2)
            kp_down_r1t2 = kp_down_r1t1 ** 2 / (nk ** 2)
            kp_down_r2t1 = kp_down_r1t1 ** 2 / (nk ** 2)
            kp_down_r2t2 = kp_down_r1t1 ** 2 / (nk ** 2)
            kp_r1t1 = kp_up_r1t1 / kp_down_r1t1
            kp_r1t2 = kp_up_r1t2 / kp_down_r1t2
            kp_r2t1 = kp_up_r2t1 / kp_down_r2t1
            kp_r2t2 = kp_up_r2t2 / kp_down_r2t2

            if abs(kp_r1t1 - kp_r2t1) <= abs(kp_r1t1 - kp_r2t2) and abs(kp_r1t2 - kp_r2t2) <= abs(kp_r1t2 - kp_r2t1):
                dis_r1t1.append(0.0514 * max2_index_r1[0, j])
                dis_r2t1.append(0.0514 * max2_index_r2[0, j])
                dis_r1t2.append(0.0514 * max2_index_r1[1, j])
                dis_r2t2.append(0.0514 * max2_index_r2[1, j])
            elif abs(kp_r1t1 - kp_r2t1) > abs(kp_r1t1 - kp_r2t2) and abs(kp_r1t2 - kp_r2t2) > abs(kp_r1t2 - kp_r2t1):
                dis_r1t1.append(0.0514 * max2_index_r1[0, j])
                dis_r2t1.append(0.0514 * max2_index_r2[1, j])
                dis_r1t2.append(0.0514 * max2_index_r1[1, j])
                dis_r2t2.append(0.0514 * max2_index_r2[0, j])
    return dis_r1t1, dis_r1t2, dis_r2t1, dis_r2t2


