from framework.evaluator import Evaluator
import framework.utils.common as utils
from tellme.data import TellmeData
import numpy as np
from time import time
import os

params = utils.load_param_file('params.py')
ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')


e = Evaluator.load2(ckpt)



training_data = TellmeData(tellme_datadir='/mnt/work/tellme/data',datafiles=['tst1.npy', 'tst2.npy'])


#  'timing_info': timing_info

correct = 0
incorrect = 0
batch_size = 4000
num_test_records = 80000 #22596471
for i in range(int(num_test_records/batch_size)):
	batch = training_data.next_batch(batch_size=batch_size)
	[r] = e.eval({'tcids_before': batch['tcids_before']}, {'sm_decision'})
	for model, gt in zip(r, batch['y']):
		model_predict =np.argmax(model)
		if model_predict == int(gt):
			correct += 1
		else:
			incorrect += 1

	print('accuracy = %0.4f'%(correct/(correct+incorrect)))

f = open('accuracy.txt', 'a')
f.write('%s %s\n'%(time(),(correct/(correct+incorrect))))
f.close()
print('accuracy = %0.4f' % (correct / (correct + incorrect)))
