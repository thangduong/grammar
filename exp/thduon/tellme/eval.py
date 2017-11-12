from framework.evaluator import Evaluator
import framework.utils.common as utils
from tellme.data import TellmeData
import numpy as np
from time import time
import gflags
import os
import sys

gflags.DEFINE_string('paramsfile', 'params.py', 'parameter files')
FLAGS = gflags.FLAGS

def eval(params):
	ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
											utils.get_dict_value(params, 'model_name') + '.ckpt')
	accuracy_file = os.path.join(utils.get_dict_value(params,'output_location'),
											'accuracy.txt')
	e = Evaluator.load2(ckpt)
	training_data = TellmeData(tellme_datadir='/mnt/work/tellme/data',datafiles=['tst1.npy', 'tst2.npy'])
	correct_list = []
	incorrect_list = []
	batch_size = 5000
#	num_test_records = 0000 #22596471
	for i in range(20):
		batch = training_data.next_batch(batch_size=batch_size)
		[r] = e.eval({'tcids_before': batch['tcids_before']}, {'sm_decision'})
		ccorrect = 0
		cincorrect = 0
		for model, gt in zip(r, batch['y']):
			model_predict =np.argmax(model)
			if model_predict == int(gt):
				ccorrect += 1
			else:
				cincorrect += 1
		correct_list.append(ccorrect)
		incorrect_list.append(cincorrect)
		print('accuracy = %0.4f'%(ccorrect/(ccorrect+cincorrect)))
	accuracy_list = [c/(c+ic) for c,ic in zip(correct_list,incorrect_list)]
	correct = np.sum(correct_list)
	incorrect = np.sum(incorrect_list)
	total_accuracy = np.mean(accuracy_list)#(correct / (correct + incorrect));
	accuracy_std = np.std(accuracy_list)
	accuracy_sem = accuracy_std / np.sqrt(len(correct_list))
	f = open(accuracy_file, 'a')
	f.write('%s %s\n'%(time(),total_accuracy))
	f.close()
	print('accuracy = %0.4f +/- %0.4f (std=%0.4f)' % (total_accuracy,accuracy_sem,accuracy_std))
	return total_accuracy, accuracy_sem, accuracy_std


def main(argv):
	try:
		argv = FLAGS(argv)  # parse flags
	except gflags.FlagsError as e:
		print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))
		sys.exit(1)
	print(FLAGS.paramsfile)
	params = utils.load_param_file(FLAGS.paramsfile)
	eval(params)


if __name__ == '__main__':
	main(sys.argv)