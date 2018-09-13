import numpy as np
import csv
lis = [['a', 'b', 'c'], [1,2,3], ['i', 'j', 'k', 'l']]
np.savetxt('test.txt', lis, delimiter = ',')
with open('test.txt', 'wb') as f:
	writer = csv.writer(f)
	writer.writerow(lis)