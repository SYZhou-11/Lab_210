import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()   #生成子图，相当于fig = plt.fi
xdata, ydata = [], []      #初始化两个数组
ln, = ax.plot([], [], 'r-', animated=False)  #第三个参数表示画曲线的颜

def init():
    ax.set_xlim(0, 2*np.pi)  #设置x轴的范围pi代表3.14...圆周率，
    ax.set_ylim(-1, 1)#设置y轴的范围
    return ln,               #返回曲线

def update(n):
    xdata.append(n)         #将每次传过来的n追加到xdata中
    ydata.append(np.sin(n))
    ln.set_data(xdata, ydata)    #重新设置曲线的值
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 10),init_func=init, blit=True)
plt.show()