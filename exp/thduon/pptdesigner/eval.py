from data import PPTDesignerData
from framework.evaluator import Evaluator
import framework.utils.common as utils
from time import time
import numpy as np
import os

model_name = 'pptrankerV0'
paramsfile = 'output/%s/params.py'%model_name
params = utils.load_param_file(paramsfile)

ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')

e = Evaluator.load2(ckpt)

test_data = PPTDesignerData(params, files=[['features_029.npy','scores_029.npy']])
model_results = []

timestr = str(int(time()))
f = open(os.path.join(utils.get_dict_value(params,'output_location'),
											'heldout_%s.txt'%timestr), 'w')
f.write('Exec Time\tModel Score\tGround Truth\tSentence\n')
for batch_no in range(10):
	print("WORKING ON BATCH %s" % batch_no)
	batch = test_data.next_batch(batch_size=10000)
	for features, ground_truth in zip(batch['features'], batch['y']):
		before_time = time()
		r = e.eval({'features': [features]}, {'sm_decision'})
		after_time = time()
		model_score = r[0][0][1]
		model_results.append([features, model_score, ground_truth])
		f.write('%0.8f\t%0.8f\t%s\n'%(after_time-before_time,model_score,ground_truth))
f.close()

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
