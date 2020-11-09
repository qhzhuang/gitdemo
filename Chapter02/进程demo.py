import threading
import time
import multiprocessing

# balance = 0
lock = threading.Lock()
balance = 0
def fun1(a, b, c):
    global balance
    for i in range(5):
        # lock.acquire()
        time.sleep(0.5)

        balance -= a
        balance += a

    # print("a:", a, "b:", "c", c)
    # print(threading.current_thread().name)
    print(threading.current_thread().name, "finished")


def fun2(a, b, c):
    global balance
    for i in range(10):
        lock.acquire()
        balance += a
        time.sleep(0.05)
        balance -= a
    # print("a:", a, "b:", "c", c)
    # print(threading.current_thread().name)
    print(threading.current_thread().name, "finished")

if __name__ == "__main__":
    # t1 = threading.Thread(target=fun1, args=(1, 2, 3))
    # t2 = threading.Thread(target=fun2, args=(1, 2, 1))
    # t1.start()
    # t2.start()
    # t1.join()
    # t2.join()
    for i in range(multiprocessing.cpu_count()):
        t = threading.Thread(target=fun1, args=(1, 2, 3))
        t.start()

        # print("total thread num", threading.enumerate())
    t.join()
    print(balance)


