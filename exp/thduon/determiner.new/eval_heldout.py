from framework.utils.data.text_indexer import TextIndexer
from word_classifier.data import ClassifierData
from framework.evaluator import Evaluator
import framework.utils.common as utils
from time import time
import numpy as np
import os

params = utils.load_param_file('params.py')

vocab_file = os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl')
ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')

e = Evaluator.load2(ckpt)
i = TextIndexer.from_file(vocab_file)

test_data = ClassifierData.get_monolingual_test(params=params)
model_results = []

timestr = str(int(time()))
f = open(os.path.join(utils.get_dict_value(params,'output_location'),
											'heldout_%s.txt'%timestr), 'w')
fe = open(os.path.join(utils.get_dict_value(params,'output_location'),
											'heldout_%s_err.txt'%timestr), 'w')
f.write('Exec Time\tModel Pick\tModel Score\tGround Truth\tSentence\n')
fe.write('Exec Time\tModel Pick\tModel Score\tGround Truth\tSentence\n')
no_right = [0,0,0,0]
no_total = [0,0,0,0]
topn = 2
for batch_no in range(1):
	print("WORKING ON BATCH %s" % batch_no)
	batch = test_data.next_batch(batch_size=20000)
	for sentence, ground_truth in zip(batch['sentence'], batch['y']):
		_, indexed, _, _ = i.index_wordlist(sentence)
		before_time = time()
		r = e.eval({'sentence': [indexed]}, {'sm_decision'})
		after_time = time()
		pick = np.argpartition(r[0][0], -topn)[-topn:]
#		pick = np.argmax(r[0][0])
		pickpr = [r[0][0][idx] for idx in pick]
		model_score = r[0][0]
		x = [after_time-before_time, pick, pickpr, ground_truth, sentence]
		if  ground_truth in pick:
			no_right[ground_truth] += 1
		else:
			fe.write('%s\n' % x)
			print([ground_truth, pick, pickpr])
			print(sentence)
		no_total[ground_truth] += 1
		model_results.append(x)
		f.write('%s\n'%x)
#for x in model_results:
#	print(x)
print("accuracy = %s"%[x/y for x,y in zip(no_right,no_total)])
#		f.write('%0.8f\t%0.8f\t%s\t%s\n'%(after_time-before_time,model_score,ground_truth,sentence))
f.close()
fe.close()