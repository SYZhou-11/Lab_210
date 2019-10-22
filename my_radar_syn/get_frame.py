# TODO 从传感器获取雷达回波帧数据

from __future__ import division, print_function
from optparse import OptionParser
from pymoduleconnector import ModuleConnector
from time import sleep
import matplotlib.pyplot as plt
import winsound
import numpy as np
import time
import cv2


def reset(device_name):
    mc = ModuleConnector(device_name)
    xep = mc.get_xep()
    xep.module_reset()
    mc.close()
    sleep(3)


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
    xep.x4driver_set_frame_area(0, 9.0)

    return xep


def record(device_name, baseband=True):
    xep = xep_setup(device_name, baseband)
    test_seconds = 10
    fps = 10   #雷达帧，走路10-50

    result, frames, meas_sampling_time = record_run(xep, fps, test_seconds, baseband)

    xep.mc.close()
    print("FPS: %g, TestTime(seconds): %d, Result: %s" % (fps, test_seconds, result))
    print("Calculated FPS based on **computer timing**: ", len(frames) / meas_sampling_time)

    return frames


def record_run(device, fps, test_seconds, baseband):
    xep = device
    nframes = test_seconds * fps
    framecounter = np.zeros(nframes)

    xep.x4driver_set_fps(fps)
    time.sleep(max(2. / fps, 5e-2))

    if xep.peek_message_data_float() == 0:
        print("FPS %d fails" % fps)
        xep.x4driver_set_fps(0)
        return False

    winsound.Beep(5000, 1000)
    frame = xep.read_message_data_float().get_data()
    frames = np.zeros((nframes, len(frame)))
    clear_buffer(xep)

    print("Start : %s" % time.ctime())
    tic = time.time()

    for icounter in range(nframes):
        d = xep.read_message_data_float()
        frames[icounter] = np.array(d.get_data())
        framecounter[icounter] = d.info

    xep.x4driver_set_fps(0)
    meas_sampling_time = time.time() - tic
    clear_buffer(xep)

    if baseband:
        n = int(len(frames[0]) / 2)
        frames_real = frames[:, :n]
        frames_imag = frames[:, n:]
        frames = frames_real + 1j * frames_imag
#       np.savetxt('test_4.8\\walk\\3\\real_10.txt', frames_real, fmt='%.9f')
#       np.savetxt('test_4.8\\walk\\3\\imag_10.txt', frames_imag, fmt='%.9f')
        np.save('2radar_2nd__{}'.format(time.time()), frames)
    else:
        np.save('rf_{}'.format(time.time()), frames)

    result = True
    check_counter = (np.diff(framecounter) == 1).all()
    # Crude check if timing of off by more than tfactor*100%
    tfactor = 0.01
    check_time = abs(test_seconds - meas_sampling_time) < tfactor * test_seconds

    result &= check_counter
    result &= check_time

    return result, frames, meas_sampling_time


def main():
    parser = OptionParser()
    parser.add_option(
        "-d",
        "--device",
        dest="device_name",
        default="com4",
        help="device file to use",
        metavar="FILE")
    parser.add_option(
        "-b",
        "--baseband",
        action="store_true",
        default=True,
        dest="baseband",
        help="Enable baseband, rf data is default")

    (options, args) = parser.parse_args()
    if not options.device_name:
        parser.error("Missing -d or -f. See --help.")
    else:
        sleep(5)

        frames = record(options.device_name, baseband=options.baseband)
        print("End : %s" % time.ctime())
        winsound.Beep(2000, 500)
        winsound.Beep(3000, 500)

        '''
        可视化原始数据和时频图
        '''
        frames1 = cv2.resize(abs(frames), (64, 64))
        plt.imshow(frames1)
        plt.show()



if __name__ == "__main__":
    main()
