import numpy as np
import math

sigma = 2
kernel_size = 11

a = np.zeros((kernel_size, kernel_size))
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        i1 = i - kernel_size // 2
        j1 = j - kernel_size // 2
        a[i, j] = math.exp(-(i1*i1+j1*j1)/(2*sigma*sigma))/(2*math.pi*sigma*sigma)

a = a / (np.sum(a))
print(np.sum(a))
print(a)
