import matplotlib.pyplot as plt
import numpy as np
#import scipy.signal as signal


def my_oscfar(arr, nn, alpha, set_n_index):
    fast_t = arr.shape[0]
    low_t = arr.shape[1]
    data_oscfar = np.zeros(shape=(fast_t, low_t))
    nn_half = int(nn / 2)
    sort_k = int(nn * 0.75)
    data_cons = np.zeros(shape=(fast_t + nn, low_t))
    data_cons[nn_half:fast_t + nn_half, 0:low_t] = arr
    for j in range(low_t):
        data_cons[0:nn_half, j] = arr[0, j]
        data_cons[fast_t + nn_half:fast_t + nn, j] = arr[fast_t - 1, j]
    for j in range(low_t):
        for i in range(fast_t):
            temp = np.hstack([data_cons[i:i + nn_half, j], data_cons[i + nn_half + 1:i + nn + 1, j]])
            temp1 = np.sort(abs(temp))
            #temp1 = np.sort(temp)
            if abs(arr[i, j]) >= temp1[sort_k - 1] * alpha:
            #if arr[i, j] >= temp1[sort_k - 1] * alpha:
                data_oscfar[i, j] = 1
    data_oscfar[0:set_n_index, ] = 0
    return data_oscfar


def my_win2d(arr, n, tn):
    fast_t = arr.shape[0]
    low_t = arr.shape[1]
    arr1 = np.zeros(shape=(fast_t, low_t))
    for i in range(n, fast_t-n):
        for j in range(n, low_t-n):
            if arr[i, j] == 1 and np.sum(arr[i-n:i+n+1, j-n:j+n+1]) >= tn:
                arr1[i, j] = 1
    return arr1


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
    #fast_t = data_in.shape[0]
    #low_t = data_in.shape[1]
    data_mti = my_mti(data_in)
    return data_in, data_mti


data_ori, mti_2p1 = get_main('D:\Working\PycharmSpace\my_radar_syc\p2_walk_radar_1_1570795371.564845.npy')


def find_2max(arr, n_left, n_right):
    fast_t = arr.shape[0]
    low_t = arr.shape[1]
    max2_arr = np.zeros(shape=(fast_t, low_t))
    arr1 = np.zeros(shape=(fast_t, low_t))
    for i in range(fast_t):
        for j in range(low_t):
            arr1[i, j] = arr[i, j]
    for j in range(low_t):
        max_index1 = np.argmax(abs(arr[:, j]))
        max2_arr[max_index1, j] = 1
        arr1[max_index1 - n_left:max_index1+n_right, j] = 0
        max_index2 = np.argmax(abs(arr1[:, j]))
        max2_arr[max_index2, j] = 1
    return max2_arr


def find2_max(arr, n_left, n_right, T):
    fast_t = arr.shape[0]
    low_t = arr.shape[1]
    arr1 = np.zeros(shape=(fast_t, low_t))
    arr2 = np.zeros(shape=(fast_t, low_t))
    for i in range(fast_t):
        for j in range(low_t):
            arr1[i, j] = arr[i, j]
    for j in range(low_t):
        max_index = []
        while np.max(abs(arr1[:, j])) >= T:
            max_index1 = np.argmax(abs(arr1[:, j]))
            #temp = arr2[max_index1, j]
            arr1[max_index1 - n_left:max_index1 + n_right, j] = 0
            #if temp >= np.max(abs(arr1[max_index1 - n_left:max_index1 + n_right, j])):
            max_index.append(max_index1)
        for index in max_index:
            arr2[index, j] = 1
    return arr2


#data_find2max = find_2max(mti_2p1, 10, 10)
data_find2max1 = find2_max(mti_2p1, 10, 10, 1e-3)
#data_find2max2 = find2_max(mti_2p1, 10, 20, 5e-4)
#data_find2max3 = find2_max(mti_2p1, 10, 30, 5e-4)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
"""
plt.figure(1)
plt.imshow(abs(data_ori), origin='lower')
plt.title('雷达原始数据')
plt.show()
"""
#fig = plt.figure()

#ax = fig.add_subplot(221)
plt.figure(1)
plt.imshow(abs(data_ori), origin='lower')
plt.title('原始图像')
#plt.show()
plt.figure(2)
plt.imshow(abs(mti_2p1), origin='lower')
plt.title('MTI')
#plt.show()
plt.figure(3)
plt.imshow(data_find2max1, origin='lower')
plt.title('轨迹')
plt.show()



