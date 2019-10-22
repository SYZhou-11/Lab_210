from multiprocessing import Process, Value
import time


def save_money(money):
    for i in range(10):
        time.sleep(0.1)
        money.value += 3
        print('save_money' + str(money.value))


def take_money(money):
    for i in range(10):
        time.sleep(0.1)
        new = money.value - 2
        print('take_money' + str(new))

# money为共享内存对象,给他一个初始值2000，类型为正型“i”
# 相当于开辟了一个空间，同时绑定值2000，
if __name__ == "__main__":
    money = Value('i', 10)

    d = Process(target=save_money, args=(money,))  # 这里面money是全局的，不写也可
    w = Process(target=take_money, args=(money,))
    d.start()
    w.start()
    d.join()
    w.join()

    print(money.value)
