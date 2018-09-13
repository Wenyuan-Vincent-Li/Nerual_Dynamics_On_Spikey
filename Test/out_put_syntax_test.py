import numpy as np
training_label = np.loadtxt('training_example.txt', delimiter = ',', dtype = int)
print training_label
rate = [20, 40]
i = 0
x = rate[1 - training_label[i]]
print x

a = b = i
print a, b
