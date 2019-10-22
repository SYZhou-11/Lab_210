# 双雷达单目标跟踪，单目标找多个最大值

from __future__ import division, print_function
import numpy as np
import sympy
from sympy.abc import x, y
import matplotlib.pyplot as plt


def top_n_arg(last_arg, current_arr1, n):
    sort_arr = np.argsort(-current_arr1)  # 得到按值降序排列对应的索引值序列
    top_arg = sort_arr[0:n]  # 取出最大的n个索引
    top_arg -= last_arg
    near_arg = np.argmin(abs(top_arg))  # 取出n个索引中距离上帧距离单元最近的
    final_arg = top_arg[near_arg] + last_arg
    return final_arg


def my_mti(arr):
    fast_t = arr.shape[0]
    low_t = arr.shape[1]
    arr1 = np.zeros(shape=(fast_t, low_t))
    for i in range(fast_t):
        for j in range(low_t - 1):
            arr1[i, j] = arr[i, j + 1] - arr[i, j]
    arr1[:, low_t - 1] = arr1[:, low_t - 2]
    return arr1


def get_main(filename):
    data_in = np.load(filename)
    data_in = abs(data_in)
    data_in = data_in.T
    data_mti = my_mti(data_in)
    return data_in, data_mti


def process_record_radar(arr):
    low_t = arr.shape[1]
    range_list = list()
    current_index = np.argmax(np.abs(arr[:, 0]))
    range_list.append(0.0514 * current_index)
    for j in range(1, low_t):
        last_index = current_index
        current_index = top_n_arg(last_index, np.abs(arr[:, j]), 6)
        if current_index - last_index >= 5:
            current_index = last_index
        range_list.append(0.0514 * current_index)
    return range_list


def get_point(rad1_list, rad2_list):
    x_list = list()
    y_list = list()
    counter_1 = 1
    counter_2 = 1
    while counter_1 < len(rad1_list) and counter_2 < len(rad2_list):
        dis1 = rad1_list[counter_1]
        dis2 = rad2_list[counter_2]
        #if dis1 > 5 and dis2 > 5:
        aa = sympy.solve([(x + 0.8) ** 2 + y ** 2 - dis1 ** 2, (x - 0.8) ** 2 + y ** 2 - dis2 ** 2], [x, y])
            # if q1.value != 0 and q2.value != 0:
            # aa = sympy.solve([x ** 2 + y ** 2 - q1.value ** 2, (x - 1.6) ** 2 + y ** 2 - q2.value ** 2], [x, y])
        result = [round(aa[0][0], 2), round(abs(aa[0][1]), 2)]
        #print("当前距离为： r1 = " + str(dis1) + "m , r2 = " + str(dis2) + "m")
        print("当前坐标为： x = " + str(result[0]) + "m , y = " + str(result[1]) + "m")
        x_list.append(result[0])
        y_list.append(result[1])
        counter_1 += 1
        counter_2 += 1
    return x_list, y_list


rad1_data_ori, rad1_data_mti = get_main('D:\Working\PycharmSpace\My_radar\my_radar_syn\walk_radar_1_time.struct_time(tm_year=2019, tm_mon=10, tm_mday=22, tm_hour=9, tm_min=39, tm_sec=40, tm_wday=1, tm_yday=295, tm_isdst=0).npy')
rad2_data_ori, rad2_data_mti = get_main('D:\Working\PycharmSpace\My_radar\my_radar_syn\walk_radar_2_time.struct_time(tm_year=2019, tm_mon=10, tm_mday=22, tm_hour=9, tm_min=39, tm_sec=40, tm_wday=1, tm_yday=295, tm_isdst=0).npy')
rad1_range_list = process_record_radar(rad1_data_mti)
rad2_range_list = process_record_radar(rad2_data_mti)
print(rad1_range_list)
print(rad2_range_list)
coord_x, coord_y = get_point(rad1_range_list, rad2_range_list)
print(coord_x)
print(coord_y)

plt.figure(1)
plt.imshow(abs(rad1_data_ori), origin='lower')
plt.title('Radar1_Ori')
plt.figure(2)
plt.imshow(abs(rad1_data_mti), origin='lower')
plt.title('Radar1_mti')
plt.figure(3)
plt.imshow(abs(rad2_data_ori), origin='lower')
plt.title('Radar2_Ori')
plt.figure(4)
plt.imshow(abs(rad2_data_mti), origin='lower')
plt.title('Radar2_mti')
plt.figure(5)
plt.axis([-3, 3, 0, 6])
ax = plt.gca()
ax.xaxis.set_ticks_position('top')
ax.yaxis.set_ticks_position('left')
ax.invert_yaxis()
ax.spines['left'].set_position(('data', 0))
plt.plot(coord_x, coord_y, 'o', markersize=7, color='blue', alpha=0.5)
plt.title('The Result')
plt.show()


