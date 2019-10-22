# TODO 目标跟踪

from __future__ import division, print_function
from pymoduleconnector import ModuleConnector
from time import sleep
import matplotlib.pyplot as plt
import winsound
import numpy as np
import time


def reset(device_name):
	mc = ModuleConnector(device_name)
	xep = mc.get_xep()
	xep.module_reset()
	mc.close()
	sleep(1)


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
		# Profile not running, OK
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
	xep.x4driver_set_frame_area(0, 5.0)

	return xep


def record(device_name, baseband=True):
	xep = xep_setup(device_name, baseband)
	test_seconds = 10
	fps = 10

	result, frames_diff, dis_record, meas_sampling_time = record_run(xep, fps, test_seconds)

	xep.mc.close()
	print("FPS: %g, TestTime(seconds): %d, Result: %s" % (fps, test_seconds, result))
	print("Calculated FPS based on **computer timing**: ", test_seconds * fps / meas_sampling_time)

	return frames_diff, dis_record


def estimate_dis(temp):
	a = temp[-3]
	b = temp[-2]
	c = temp[-1]

	if abs(b - a) >= 10 and abs(c - b) < 10:
		distance = int(round((b + c) / 2))
		temp[-3] = b

	elif abs(b - a) < 10 and abs(c - b) >= 10:
		distance = int(round((a + b) / 2))
		temp[-1] = b

	elif abs(b - a) < 10 and abs(c - b) < 10:
		distance = int(round((a + b + c) / 3))

	else:
		distance = int(round((a + c) / 2))
		temp[-2] = distance

	return distance


def record_run(device, fps, test_seconds):
	xep = device
	nframes = test_seconds * fps  # 慢时间
	framecounter = np.zeros(nframes)

	xep.x4driver_set_fps(fps)
	time.sleep(max(2. / fps, 5e-2))

	if xep.peek_message_data_float() == 0:
		print("FPS %d fails" % fps)
		xep.x4driver_set_fps(0)
		raise Exception("False")

	winsound.Beep(5000, 1000)
	frame = xep.read_message_data_float().get_data()
	lframe = int(len(frame) / 2)  # 采样时实部和虚部分开采集，所以快时间=总长/2
	frames = np.zeros((nframes, lframe), dtype=np.complex64)
	frames_diff = np.zeros((nframes - 1, lframe), dtype=np.complex64)
	dis_max = list()
	dis_record = list()

	clear_buffer(xep)

	print("Start : %s" % time.ctime())
	tic = time.time()

	for icounter in range(nframes):
		frame = xep.read_message_data_float().get_data()
		frames[icounter] = np.array(frame[:lframe]) + 1j * np.array(frame[lframe:])

		if icounter > 0:
			frames_diff[icounter - 1] = frames[icounter] - frames[icounter - 1]
			dis_max.append(np.argmax(np.abs(frames_diff[icounter - 1])))
		if len(dis_max) >= 3:
			distance = estimate_dis(dis_max)
			if np.abs(frames[icounter - 3, dis_max[icounter-3]]) < 0.0015 \
					and np.abs(frames[icounter - 2, dis_max[icounter-2]]) < 0.0015 \
					and np.abs(frames[icounter - 1, dis_max[icounter-1]]) < 0.0015:
				dis_record.append(0)
				print("[INFO] No target")
			else:
				dis_record.append(0.0514 * distance)
				print("[INFO] Target distance = %.1f m" % (0.0514 * distance))


	xep.x4driver_set_fps(0)
	meas_sampling_time = time.time() - tic
	clear_buffer(xep)

	np.save('walk_{}'.format(time.time()), frames)

	result = True
	# Crude check if timing of off by more than tfactor*100%
	tfactor = 0.01
	check_time = abs(test_seconds - meas_sampling_time) < tfactor * test_seconds
	result &= check_time

	return result, frames_diff, dis_record, meas_sampling_time


def main():
	com = 'com3' # com口
	frames_diff, dis_record = record(com)
	print("End : %s" % time.ctime())
	winsound.Beep(2000, 500)
	winsound.Beep(3000, 500)

	plt.plot(np.array(range(np.shape(frames_diff)[0] - 2)) + 1, np.array(dis_record), '*')
	plt.plot(np.array(range(np.shape(frames_diff)[0])), 0.0514 * np.argmax(np.abs(frames_diff), axis=1))
	plt.grid()
	plt.xlabel("Frames")
	plt.ylabel("Distance (m)")
	plt.show()


if __name__ == "__main__":
	main()
