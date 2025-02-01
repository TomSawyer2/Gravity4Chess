import numpy as np

times = [
    8.065,
    5.064,
    3.554,
    1.491,
    1.209,
    6.911,
    4.319,
    5.454,
    3.984,
    0,
    0,
    4.83,
    0,
    0,
    0,
    0,
    8.192,
    5.602,
    6.081,
    8.907,
    6.564,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0.059,
    0.00200009,
    0,
]

print("共计算了{}步".format(len(times)))
print("总用时：{}s".format(sum(times)))
print("平均用时：{}s".format(sum(times) / len(times)))
print("最大用时：{}s".format(max(times)))
print("最小用时：{}s".format(min(times)))
print("用时标准差：{}".format(np.std(times)))
print("用时方差：{}".format(np.var(times)))
print("用时中位数：{}".format(np.median(times)))
