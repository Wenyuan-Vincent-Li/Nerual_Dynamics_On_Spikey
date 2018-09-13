import numpy as np 

data0 = np.loadtxt('lambda_parameter_0.txt', delimiter = ',')
data1 = np.loadtxt('lambda_parameter_1.txt', delimiter = ',')
data = np.vstack((data0, data1))
data = data[np.random.permutation(data.shape[0]), :]
training_data = data[:int(np.size(data, 0) * 0.6), :]
testing_data = data[int(np.size(data, 0) * 0.6) : , :]
np.savetxt('training_data_0_1.txt', training_data, delimiter = ',')
np.savetxt('testing_data_0_1.txt', testing_data, delimiter = ',')
# print np.size(training_data, 0)
# print np.size(testing_data, 0)
# print np.size(training_data, 0) + np.size(testing_data, 0)