import numpy as np
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
    return data_in, data_mti


check_radar1_ori, check_radar1_mti = get_main('D:\Working\PyCharmSpace\my_radar\My_radar\my_radar_syn\outside_p2_walk_radar_1_1571984691.5356767.npy')
check_radar2_ori, check_radar2_mti = get_main('D:\Working\PyCharmSpace\my_radar\My_radar\my_radar_syn\outside_p2_walk_radar_2_1571984691.5356767.npy')
plt.figure(1)
plt.imshow(abs(check_radar1_ori), origin='lower')
plt.title('Radar1_Ori')
plt.figure(2)
plt.imshow(abs(check_radar1_mti), origin='lower')
plt.title('Radar1_mti')
plt.figure(3)
plt.imshow(abs(check_radar2_ori), origin='lower')
plt.title('Radar2_Ori')
plt.figure(4)
plt.imshow(abs(check_radar2_mti), origin='lower')
plt.title('Radar2_mti')
plt.show()