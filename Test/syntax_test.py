# b = 1
# a = b ==1
# print (['WRONG', 'CORRECT'][int(a)])

import sys
import time
import logging
import numpy

sys.path.append('/Users/wenyuan/Documents/PyNN')
logging.basicConfig()
lg = logging.getLogger('test_loging')
lg.setLevel(logging.INFO)
try:
	import pyNN.brian as sim
except ImportError, e:
	print('ImportError:{}.\n'.format(e.message))
	sys.exit(1)

import neuclar.network_controller as netcontrol

from DLSM_test.network_config import software_example_config as config
# print config['network']['input_source_1']
# # config['network']['decision_pops'] = '{}'.format(5)
# # print config['network']['decision_pops']
# # lg.info('performing learning')
# # lg.info('Classifier: %s %s -> %s - - %s.' %(15, 5, 10, ['Wrong', 'Correct'][1]))
# # lg.debug('class_ids:5')
