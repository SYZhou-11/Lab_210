# 双雷达单目标跟踪，单目标找多个最大值

from __future__ import division, print_function
from pymoduleconnector import ModuleConnector
import winsound
import numpy as np
import time, os
import sympy
from sympy.abc import x, y
from multiprocessing import Process, Manager, Pool, Queue

test_seconds = 10
fps = 10
nframes = test_seconds * fps


def reset(device_name):
	mc = ModuleConnector(device_name)
	xep = mc.get_xep()
	xep.module_reset()
	mc.close()
	time.sleep(3)


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
	xep.x4driver_set_frame_area(0, 6.0)

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
	near_arg = np.argmin(abs(top_arg))
	final_arg = top_arg[near_arg] + last_arg
	return final_arg


def record_radar(device_name, i, data_queue1):
	#global xep_rad1
	xep_rad1 = xep_setup(device_name, baseband=True)
	xep_rad1.x4driver_set_fps(fps)
	time.sleep(max(2. / fps, 5e-2))

	if xep_rad1.peek_message_data_float() == 0:
		print("FPS %d fails" % fps)
		xep_rad1.x4driver_set_fps(0)
		raise Exception("False")

	#winsound.Beep(5000, 1000)
	frame = xep_rad1.read_message_data_float().get_data()

	#global lframe, frames_diff_rad1, frames_rad1, dis_max_rad1, dis_record_rad1, icounter_rad1, temp_icounter_rad1

	lframe = int(len(frame) / 2)  # 采样时实部和虚部分开采集，所以快时间=总长/2
	frames_rad1 = np.zeros((nframes, lframe), dtype=np.complex64)
	frames_diff_rad1 = np.zeros((nframes - 1, lframe), dtype=np.complex64)
	dis_max_rad1 = list()
	#dis_record_rad1 = list()

	clear_buffer(xep_rad1)

	for icounter_rad1 in range(nframes):
		frame = xep_rad1.read_message_data_float().get_data()
		frames_rad1[icounter_rad1] = np.array(frame[:lframe]) + 1j * np.array(frame[lframe:])
		if icounter_rad1 > 0:
			frames_diff_rad1[icounter_rad1 - 1] = frames_rad1[icounter_rad1] - frames_rad1[icounter_rad1 - 1]
			if icounter_rad1 == 1:
				last_index = np.argmax(np.abs(frames_diff_rad1[icounter_rad1 - 1]))
				dis_max_rad1.append(last_index)
			else:
				last_index = dis_max_rad1[-1]
				dis_max_rad1.append(top_n_arg(last_index, np.abs(frames_diff_rad1[icounter_rad1 - 1]), 5))
			real_dis = 0.0514 * last_index
			print('radar_' + i + ': ', real_dis)
			data_queue1.put(real_dis)
			#time.sleep(0.5)

	xep_rad1.x4driver_set_fps(0)
	clear_buffer(xep_rad1)
	file_str = 'walk_radar_' + i + '_{}'
	np.save(file_str.format(time.time()), frames_rad1)
	xep_rad1.mc.close()


def get_point(q1, q2):
	#while not q1.empty() and not q2.empty():
	while True:
		dis1 = q1.get(True)
		dis2 = q2.get(True)
		aa = sympy.solve([(x + 0.8) ** 2 + y ** 2 - dis1 ** 2, (x - 0.8) ** 2 + y ** 2 - dis2 ** 2], [x, y])
		result = [round(aa[0][0], 2), round(abs(aa[0][1]), 2)]
		print("当前距离为： r1 = " + str(dis1) + "m , r2 = " + str(dis2) + "m")
		print("当前坐标为： x = " + str(result[0]) + "m , y = " + str(result[1]) + "m")


if __name__ == "__main__":
	radar_q1 = Queue()
	radar_q2 = Queue()
	pg_rad1 = Process(target=record_radar, args=('COM3', '1', radar_q1,))
	pg_rad2 = Process(target=record_radar, args=('COM4', '2', radar_q2,))
	p_out = Process(target=get_point, args=(radar_q1, radar_q2,))

	pg_rad1.start()
	#time.sleep(1)
	pg_rad2.start()
	p_out.start()
	pg_rad1.join()
	pg_rad2.join()
	p_out.terminate()
	"""
	pool = Pool(3)
	pool.apply_async(record_radar, ('COM5', '1', radar_q1,))
	pool.apply_async(record_radar, ('COM4', '2', radar_q2,))
	pool.apply_async(get_point, (radar_q1, radar_q2,))
	#pool.close()
	#pool.join()
	"""



