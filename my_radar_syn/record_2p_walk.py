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
    while xep.peek_message_data_float():
        xep.read_message_data_float()


def xep_setup(device_name, baseband):
    reset(device_name)
    mc = ModuleConnector(device_name)
    app = mc.get_x4m300()
    try:
        app.set_sensor_mode(0x13, 0)  # Make sure no profile is running.
    except RuntimeError:
        pass
    try:
        app.set_sensor_mode(0x12, 0)  # Manual mode.
    except RuntimeError:
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


def record_radar(device_name, i):
    # global xep_rad1
    xep_rad1 = xep_setup(device_name, baseband=True)
    xep_rad1.x4driver_set_fps(fps)
    time.sleep(max(2. / fps, 5e-2))

    if xep_rad1.peek_message_data_float() == 0:
        print("FPS %d fails" % fps)
        xep_rad1.x4driver_set_fps(0)
        raise Exception("False")

    # winsound.Beep(5000, 1000)
    frame = xep_rad1.read_message_data_float().get_data()

    # global lframe, frames_diff_rad1, frames_rad1, dis_max_rad1, dis_record_rad1, icounter_rad1, temp_icounter_rad1

    lframe = int(len(frame) / 2)  # 采样时实部和虚部分开采集，所以快时间=总长/2
    frames_rad1 = np.zeros((nframes, lframe), dtype=np.complex64)
    frames_diff_rad1 = np.zeros((nframes - 1, lframe), dtype=np.complex64)
    dis_max_rad1 = list()
    # dis_record_rad1 = list()

    clear_buffer(xep_rad1)

    for icounter_rad1 in range(nframes):
        frame = xep_rad1.read_message_data_float().get_data()
        frames_rad1[icounter_rad1] = np.array(frame[:lframe]) + 1j * np.array(frame[lframe:])

        if icounter_rad1 > 0:
            frames_diff_rad1[icounter_rad1 - 1] = frames_rad1[icounter_rad1] - frames_rad1[icounter_rad1 - 1]
            dis_max_rad1.append(np.argmax(np.abs(frames_diff_rad1[icounter_rad1 - 1])))
        if len(dis_max_rad1) >= 3:
            distance = estimate_dis(dis_max_rad1, 10)
            # dis_record_rad1.append(0.0514 * distance)
            real_dis = 0.0514 * distance
            print('radar_' + i + ': ', real_dis)
        # time.sleep(0.5)

    xep_rad1.x4driver_set_fps(0)
    clear_buffer(xep_rad1)
    file_str = 'p2_walk_radar_' + i + '_{}'
    np.save(file_str.format(time.time()), frames_rad1)
    xep_rad1.mc.close()


if __name__ == "__main__":
    pg_rad1 = Process(target=record_radar, args=('COM5', '1',))
    pg_rad2 = Process(target=record_radar, args=('COM4', '2',))
    pg_rad1.start()
    pg_rad2.start()
    pg_rad1.join()
    pg_rad2.join()