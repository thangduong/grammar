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
ferr = open(os.path.join(utils.get_dict_value(params,'output_location'),
											'heldout_%s_err.txt'%timestr), 'w')
f.write('Exec Time\tModel Score\tGround Truth\tSentence\n')
for batch_no in range(4):
	print("WORKING ON BATCH %s" % batch_no)
	batch = test_data.next_batch(batch_size=5000)
	for sentence, ground_truth in zip(batch['sentence'], batch['y']):
		_, indexed, _, _ = i.index_wordlist(sentence)
		before_time = time()
		r = e.eval({'sentence': [indexed]}, {'sm_decision'})
		after_time = time()
		model_score = r[0][0][1]
		model_results.append([indexed, model_score, ground_truth])
		if (model_score < .5 and ground_truth > .5) or \
			(model_score > .5 and ground_truth < .5):
			s1 = [x for x in sentence[0:20] if x!='<pad>']
			s2 = [x for x in sentence[20:] if x!='<pad>']
			ferr.write('%0.8f\t%0.8f\t%s\t%s\n'%(after_time-before_time,model_score,ground_truth,"%s ___ %s"%(' '.join(s1), ' '.join(s2))))
		f.write('%0.8f\t%0.8f\t%s\t%s\n'%(after_time-before_time,model_score,ground_truth,sentence))
f.close()
ferr.close()

f = open(os.path.join(utils.get_dict_value(params,'output_location'),
											timestr + ".txt"), 'w')
for thres in np.linspace(0,1,100):
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	#	print('%s - %s'%(model_score,ground_truth))
	for indexed, model_score, ground_truth in model_results:
		if ground_truth < .5:
			if model_score > thres:
				fp += 1
			else:
				tn += 1
		else:
			if model_score > thres:
				tp += 1
			else:
				fn += 1
#	print([tp,fp,tn,fn])
	sp = tp + fp
	if sp == 0:
		sp = 1
	nr = tp + fn
	if nr == 0:
		nr = 1
	precision = tp / sp
	recall = tp / nr
#	print("precision: %s"%precision)
#	print("recall: %s"%recall)
	msg = ('%s,%s,%s'%(thres,precision,recall))
	f.write('%s\n'%msg)
	print(msg)

f.close()