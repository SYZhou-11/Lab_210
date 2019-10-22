import matplotlib.pyplot as plt
import numpy as np
import time
from math import *

plt.ion() #开启interactive mode 成功的关键函数
plt.figure(1)
t = [0]
t_now = 0
m = [sin(t_now)]

for i in range(2000):
    #plt.clf()
    t_now = i*0.1
    #t.append(t_now)
    #m.append(sin(t_now))
    plt.plot(t_now,sin(t_now),'-r')
    #plt.pause(0.01)


