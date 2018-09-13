import numpy as np 

# import os.path
# print not (os.path.isfile('input1_output1_test.txt') and os.path.isfile('input1_output2_test.txt') and os.path.isfile('input2_output2_test.txt'))
# a = np.arange(15).reshape(3, 5)
# print a
j = 0
for i in range(10):
	j += 1
	if j == 5:
		if i == 4:
			break


print i, j 





# try:
# 	np.loadtxt('input1_output5_test.txt')
# except IOError:
# 	print 1

# b = 1
# a = c = d = b
# print a, c, d
		
# res = ['E', 'I']
# dec = [0, 1]

# print ('res%s_output%d.txt'%(res[0], dec[0]))
# pre_rates = [156.0, 152.0, 66.0, 122.0, 194.0, 234.0]
# pre_rates = np.asarray(pre_rates)
# units_sortidx = np.argsort(pre_rates)
# print pre_rates[units_sortidx]
# w_max = 3
# units_sortidx = np.array([1, 2, 5, 8])
# exc = units_sortidx[units_sortidx < 4]
# print exc
# b = np.nan
# a = np.array([[1, 2], [3, b], [5, 6]])
# print np.size(a, 0)
# a[exc - 1, :] += 0.01
# # a[a > w_max] = w_max
# print a