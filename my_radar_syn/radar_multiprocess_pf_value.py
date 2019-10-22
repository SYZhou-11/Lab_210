# 双雷达单目标跟踪，单目标找多个最大值

from __future__ import division, print_function
from pymoduleconnector import ModuleConnector
import winsound
import numpy as np
import time, os
import sympy
from sympy.abc import x, y
from multiprocessing import Process, Manager, Pool, Queue
import matplotlib.pyplot as plt
import multiprocessing

test_seconds = 12
fps = 10
nframes = test_seconds * fps


def reset(device_name):
	mc = ModuleConnector(device_name)
	xep = mc.get_xep()
	xep.module_reset()
	mc.close()
	time.sleep(1)


def clear_buffer(xep):
	"""Clears the frame buffer"""
	while xep.peek_message_data_float():
		xep.read_message_data_float()


def xep_setup(device_name, baseband):
	reset(device_name)
	mc = ModuleConnector(device_name)

	# Assume an X4M300/X4M200 module and try to enter XEP mode
	app = mc.get_x4m300()
	# Stop running application and set module in manual mode.
	try:
		app.set_sensor_mode(0x13, 0)  # Make sure no profile is running.
	except RuntimeError:
		#Profile not running, OK
		pass

	try:
		app.set_sensor_mode(0x12, 0)  # Manual mode.
	except RuntimeError:
		# Sensor already stopped, OK
		pass

	xep = mc.get_xep()
	xep.mc = mc
	# Set DAC range
	xep.x4driver_set_dac_min(949)
	xep.x4driver_set_dac_max(1100)

	# Set integration
	xep.x4driver_set_iterations(16)
	xep.x4driver_set_pulses_per_step(10)
	xep.x4driver_set_downconversion(int(baseband))

	# Set detection range
	xep.x4driver_set_frame_area(0, 9.0)

	return xep


def estimate_dis(temp, flag):
	a = temp[-3]
	b = temp[-2]
	c = temp[-1]

	if abs(b - a) >= flag and abs(c - b) < flag:
		distance = int(round((b + c) / 2))
		temp[-3] = b

	elif abs(b - a) < flag and abs(c - b) >= flag:
		distance = int(round((a + b) / 2))
		temp[-1] = b

	elif abs(b - a) < flag and abs(c - b) < flag:
		distance = int(round((a + b + c) / 3))

	else:
		distance = int(round((a + c) / 2))
		temp[-2] = distance

	return distance


def top_n_arg(last_arg, current_arr1, n):
	sort_arr = np.argsort(-current_arr1) #得到按值降序排列对应的索引值序列
	top_arg = sort_arr[0:n] #取出最大的n个索引
	top_arg -= last_arg
	near_arg = np.argmin(abs(top_arg)) #取出n个索引中距离上帧距离单元最近的
	final_arg = top_arg[near_arg] + last_arg
	return final_arg


def record_radar1(device_name, i, distance1, flag1):
	#global xep_rad1
	xep_rad1 = xep_setup(device_name, baseband=True)
	xep_rad1.x4driver_set_fps(fps)
	time.sleep(max(2. / fps, 5e-2))

	if xep_rad1.peek_message_data_float() == 0:
		print("FPS %d fails" % fps)
		xep_rad1.x4driver_set_fps(0)
		raise Exception("False")

	winsound.Beep(1000, 500)
	frame = xep_rad1.read_message_data_float().get_data()

	#global lframe, frames_diff_rad1, frames_rad1, dis_max_rad1, dis_record_rad1, icounter_rad1, temp_icounter_rad1

	lframe = int(len(frame) / 2)  # 采样时实部和虚部分开采集，所以快时间=总长/2
	frames_rad1 = np.zeros((nframes, lframe), dtype=np.complex64)
	frames_diff_rad1 = np.zeros((nframes - 1, lframe), dtype=np.complex64)
	#dis_max_rad1 = list()
	#dis_record_rad1 = list()

	clear_buffer(xep_rad1)
	last_index, current_index = 0, 0

	for icounter_rad1 in range(nframes):
		frame = xep_rad1.read_message_data_float().get_data()
		frames_rad1[icounter_rad1] = np.array(frame[:lframe]) + 1j * np.array(frame[lframe:])
		if icounter_rad1 > 0:
			frames_diff_rad1[icounter_rad1 - 1] = frames_rad1[icounter_rad1] - frames_rad1[icounter_rad1 - 1]
			if icounter_rad1 == 1:
				current_index = np.argmax(np.abs(frames_diff_rad1[icounter_rad1 - 1]))
				#dis_max_rad1.append(last_index)
			else:
				last_index = current_index
				current_index = top_n_arg(last_index, np.abs(frames_diff_rad1[icounter_rad1 - 1]), 5)
				"""
				if abs(temp - last_index) <= 10:
					last_index = temp
				"""
				#last_index = temp
				#dis_max_rad1.append(last_index)
			real_dis = 0.0514 * last_index
			print('radar_' + i + ': ', real_dis)
			distance1.value = real_dis
		flag1.value = icounter_rad1
			#time.sleep(0.5)

	xep_rad1.x4driver_set_fps(0)
	clear_buffer(xep_rad1)
	file_str = 'walk_radar_' + i + '_{}'
	np.save(file_str.format(time.localtime()), frames_rad1)
	xep_rad1.mc.close()


def record_radar2(device_name, i, distance2, flag2):
	#global xep_rad1
	xep_rad1 = xep_setup(device_name, baseband=True)
	xep_rad1.x4driver_set_fps(fps)
	time.sleep(max(2. / fps, 5e-2))

	if xep_rad1.peek_message_data_float() == 0:
		print("FPS %d fails" % fps)
		xep_rad1.x4driver_set_fps(0)
		raise Exception("False")

	winsound.Beep(1000, 500)
	frame = xep_rad1.read_message_data_float().get_data()

	#global lframe, frames_diff_rad1, frames_rad1, dis_max_rad1, dis_record_rad1, icounter_rad1, temp_icounter_rad1

	lframe = int(len(frame) / 2)  # 采样时实部和虚部分开采集，所以快时间=总长/2
	frames_rad1 = np.zeros((nframes, lframe), dtype=np.complex64)
	frames_diff_rad1 = np.zeros((nframes - 1, lframe), dtype=np.complex64)
	#dis_max_rad1 = list()
	#dis_record_rad1 = list()

	clear_buffer(xep_rad1)
	last_index, current_index = 0, 0

	for icounter_rad1 in range(nframes):
		frame = xep_rad1.read_message_data_float().get_data()
		frames_rad1[icounter_rad1] = np.array(frame[:lframe]) + 1j * np.array(frame[lframe:])
		if icounter_rad1 > 0:
			frames_diff_rad1[icounter_rad1 - 1] = frames_rad1[icounter_rad1] - frames_rad1[icounter_rad1 - 1]
			if icounter_rad1 == 1:
				current_index = np.argmax(np.abs(frames_diff_rad1[icounter_rad1 - 1]))
				#dis_max_rad1.append(last_index)
			else:
				last_index = current_index
				current_index = top_n_arg(last_index, np.abs(frames_diff_rad1[icounter_rad1 - 1]), 5)
				"""
				if abs(temp - last_index) <= 10:
					last_index = temp
				"""
				#last_index = temp
				#dis_max_rad1.append(last_index)
			real_dis = 0.0514 * last_index
			print('radar_' + i + ': ', real_dis)
			distance2.value = real_dis
		flag2.value = icounter_rad1
			#time.sleep(0.5)

	xep_rad1.x4driver_set_fps(0)
	clear_buffer(xep_rad1)
	file_str = 'walk_radar_' + i + '_{}'
	np.save(file_str.format(time.localtime()), frames_rad1)
	xep_rad1.mc.close()


def get_point(distance1, distance2, flag1, flag2):
	x_list = list()
	y_list = list()
	plt.ion()
	plt.figure(1)
	while True:
		if distance1.value != 0 and distance2.value != 0:
			dis1 = distance1.value
			dis2 = distance2.value
			aa = sympy.solve([(x + 0.8) ** 2 + y ** 2 - dis1 ** 2, (x - 0.8) ** 2 + y ** 2 - dis2 ** 2], [x, y])
			# if q1.value != 0 and q2.value != 0:
			# aa = sympy.solve([x ** 2 + y ** 2 - q1.value ** 2, (x - 1.6) ** 2 + y ** 2 - q2.value ** 2], [x, y])
			result = [round(aa[0][0], 2), round(abs(aa[0][1]), 2)]
			print("当前距离为： r1 = " + str(dis1) + "m , r2 = " + str(dis2) + "m")
			# print("当前坐标为： x = " + str(result[0]) + "m , y = " + str(result[1]) + "m")
			x_list.append(result[0])
			y_list.append(result[1])
			plt.clf()
			plt.axis([-3, 3, 0, 6])
			ax = plt.gca()
			ax.xaxis.set_ticks_position('top')
			ax.yaxis.set_ticks_position('left')
			ax.invert_yaxis()
			ax.spines['left'].set_position(('data', 0))
			plt.plot(x_list, y_list, 'o', markersize=7, color='blue', alpha=0.5)
			plt.pause(0.0001)
			if flag1.value == nframes - 1 or flag2.value == nframes - 1:
				plt.ioff()
				plt.show()
				break


if __name__ == "__main__":
	#radar_q1 = Queue()
	#radar_q2 = Queue()
	flag1 = multiprocessing.Value("i", 0)  #设置为整型常数0
	flag2 = multiprocessing.Value("i", 0)
	distance1 = multiprocessing.Value("f", 0)  #设置为浮点常数0
	distance2 = multiprocessing.Value("f", 0)
	pg_rad1 = Process(target=record_radar1, args=('COM3', '1', distance1, flag1))
	pg_rad2 = Process(target=record_radar2, args=('COM4', '2', distance2, flag2))
	p_out = Process(target=get_point, args=(distance1, distance2, flag1, flag2))
	time.sleep(5)
	pg_rad1.start()
	#time.sleep(1)
	pg_rad2.start()
	p_out.start()
	pg_rad1.join()
	pg_rad2.join()
	#p_out.terminate()
	p_out.join()



