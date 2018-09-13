import random as rm
import numpy as np
import csv
filename = 'training_example.txt' # define the txt filename for storing the data
No_data = 200 # define how many data it will generate

label = [rm.randint(0, 1) for i in range(No_data)]
np.savetxt(filename, label, fmt = '%d', delimiter = ',')