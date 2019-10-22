from __future__ import division, print_function
from pymoduleconnector import ModuleConnector
import winsound
import numpy as np
import time

from multiprocessing import Process, Manager, Pool, Queue

test_seconds = 20
fps = 10
nframes = test_seconds * fps


def reset(device_name):
    mc = ModuleConnector(device_name)
    xep = mc.get_xep()
    xep.module_reset()
    mc.close()
    time.sleep(1)


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
    xep.x4driver_set_frame_area(0, 9.0)

    return xep


def record_radar(device_name, i):
    xep_rad1 = xep_setup(device_name, baseband=True)
    xep_rad1.x4driver_set_fps(fps)
    time.sleep(max(2. / fps, 5e-2))

    if xep_rad1.peek_message_data_float() == 0:
        print("FPS %d fails" % fps)
        xep_rad1.x4driver_set_fps(0)
        raise Exception("False")

    winsound.Beep(1000, 500)
    frame = xep_rad1.read_message_data_float().get_data()

    lframe = int(len(frame) / 2)  # 采样时实部和虚部分开采集，所以快时间=总长/2
    frames_rad1 = np.zeros((nframes, lframe), dtype=np.complex64)

    clear_buffer(xep_rad1)

    for icounter_rad1 in range(nframes):
        frame = xep_rad1.read_message_data_float().get_data()
        frames_rad1[icounter_rad1] = np.array(frame[:lframe]) + 1j * np.array(frame[lframe:])

    xep_rad1.x4driver_set_fps(0)
    clear_buffer(xep_rad1)
    file_str = 'p2_walk_radar_' + i + '_{}'
    np.save(file_str.format(time.localtime()), frames_rad1)
    xep_rad1.mc.close()


if __name__ == "__main__":
    pg_rad1 = Process(target=record_radar, args=('COM3', '1',))
    pg_rad2 = Process(target=record_radar, args=('COM4', '2',))
    time.sleep(5)
    pg_rad1.start()
    pg_rad2.start()
    pg_rad1.join()
    pg_rad2.join()