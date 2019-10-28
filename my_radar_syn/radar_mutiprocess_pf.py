# 双雷达单目标跟踪（实时）

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

test_seconds = 30
fps = 5
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


def top_n_arg(last_arg, current_arr1, n):
	sort_arr = np.argsort(-current_arr1) #得到按值降序排列对应的索引值序列
	top_arg = sort_arr[0:n] #取出最大的n个索引
	top_arg -= last_arg
	near_arg = np.argmin(abs(top_arg)) #取出n个索引中距离上帧距离单元最近的
	final_arg = top_arg[near_arg] + last_arg
	return final_arg


def record_radar(device_name, i, distance):
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
	last_index, current_index = 66, 66

	for icounter_rad1 in range(nframes):
		frame = xep_rad1.read_message_data_float().get_data()
		frames_rad1[icounter_rad1] = np.array(frame[:lframe]) + 1j * np.array(frame[lframe:])
		if icounter_rad1 > 0:
			frames_diff_rad1[icounter_rad1 - 1] = frames_rad1[icounter_rad1] - frames_rad1[icounter_rad1 - 1]
			if icounter_rad1 == 1:
				current_index = 66 #起点设置在坐标（0，3.2）
				#current_index = np.argmax(np.abs(frames_diff_rad1[icounter_rad1 - 1]))
				#dis_max_rad1.append(last_index)
			else:
				last_index = current_index
				current_index = top_n_arg(last_index, np.abs(frames_diff_rad1[icounter_rad1 - 1]), 6)
				if abs(current_index - last_index) >= 8:
					current_index = last_index
				#last_index = temp
				#dis_max_rad1.append(last_index)

			real_dis = 0.0514 * current_index
			print('radar_' + i + ': ', real_dis)
			distance.put(real_dis)
		#counter_flag.value = icounter_rad1
			#time.sleep(0.5)

	xep_rad1.x4driver_set_fps(0)
	clear_buffer(xep_rad1)
	file_str = 'outside_walk_radar_' + i + '_{}'
	np.save(file_str.format(time.localtime()), frames_rad1)
	xep_rad1.mc.close()


def get_point(d1, d2, flag_1, flag_2):
	x_list = list()
	y_list = list()
	plt.ion()
	plt.figure(1)
	while flag_1.value != nframes - 1 and flag_2.value != nframes - 1:
		dis1 = d1.get(True)
		dis2 = d2.get(True)
		aa = sympy.solve([(x + 0.8) ** 2 + y ** 2 - dis1 ** 2, (x - 0.8) ** 2 + y ** 2 - dis2 ** 2], [x, y])
		#雷达1(-0.8, 0) 雷达2(0.8, 0)
		result = [round(aa[0][0], 2), round(abs(aa[0][1]), 2)]
		print("当前距离为： r1 = " + str(dis1) + "m , r2 = " + str(dis2) + "m")
			#print("当前坐标为： x = " + str(result[0]) + "m , y = " + str(result[1]) + "m")
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
		flag_1.value += 1
		flag_2.value += 1
	plt.ioff()
	plt.show()



if __name__ == "__main__":
	radar_q1 = Queue()
	radar_q2 = Queue()
	flag1 = multiprocessing.Value("i", 0)  #设置为整型常数0
	flag2 = multiprocessing.Value("i", 0)
	#distance1 = multiprocessing.Value("f", 0)  #设置为浮点常数0
	#distance2 = multiprocessing.Value("f", 0)
	pg_rad1 = Process(target=record_radar, args=('COM3', '1', radar_q1))
	pg_rad2 = Process(target=record_radar, args=('COM4', '2', radar_q2))
	p_out = Process(target=get_point, args=(radar_q1, radar_q2, flag1, flag2))

	time.sleep(5)

	pg_rad1.start()
	pg_rad2.start()
	p_out.start()
	pg_rad1.join()
	pg_rad2.join()
	winsound.Beep(1000, 500)
	#p_out.terminate()
	p_out.join()



