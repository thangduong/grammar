from framework.utils.data.text_indexer import TextIndexer
from word_classifier.data import ClassifierData
from framework.evaluator import Evaluator
import framework.utils.common as utils
from time import time
import numpy as np
import os

params = utils.load_param_file('output/determinerV3/params.py')

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
fip = open(os.path.join(utils.get_dict_value(params,'output_location'),
											'heldout_%s_err2.txt'%timestr), 'w')
f.write('Exec Time\tModel Pick\tModel Score\tGround Truth\tSentence\n')
fe.write('Exec Time\tModel Pick\tModel Score\tGround Truth\tSentence\n')
no_right = [0,0,0,0]
no_total = [0,0,0,0]
no_total_model = [0,0,0,0]
error_scenario = [[0,0,0,0],
									[0, 0, 0, 0],
									[0, 0, 0, 0],
									[0, 0, 0, 0]]
topn = 1
last = 0
for batch_no in range(1):
	print("WORKING ON BATCH %s" % batch_no)
	batch = test_data.next_batch(batch_size=200000)
	for idx, (sentence, ground_truth) in enumerate(zip(batch['sentence'], batch['y'])):
		_, indexed, _, _ = i.index_wordlist(sentence)
		before_time = time()
		r = e.eval({'sentence': [indexed]}, {'sm_decision'})
		after_time = time()
		pick = np.argpartition(r[0][0], -topn)[-topn:]
#		pick = np.argmax(r[0][0])
		pickpr = [r[0][0][idx] for idx in pick]
		model_score = r[0][0]
		x = [after_time-before_time, pick, pickpr, ground_truth, sentence]
		if ground_truth in pick:
			no_right[ground_truth] += 1
			fip.write('1 %s %s\n'%(ground_truth,pickpr[0]))
		else:
			fip.write('0 %s %s\n'%(ground_truth,pickpr[0]))
			fe.write('%s\n' % x)
#			print([idx, ground_truth, pick, pickpr])
#			print(sentence)
		error_scenario[ground_truth][pick[0]] += 1
		no_total[ground_truth] += 1
		no_total_model[pick[0]] += 1
		model_results.append(x)
		f.write('%s\n'%x)
		if idx-last > 10000:
			print("recall = %s" % [x / y for x, y in zip(no_right, no_total)])
			print("precision = %s" % [x / y for x, y in zip(no_right, no_total_model)])
			print(no_right)
			print(no_total)
			print(no_total_model)
			print(error_scenario)
			last =idx

#for x in model_results:
#	print(x)
print("recall = %s"%[x/y for x,y in zip(no_right,no_total)])
print("precision = %s"%[x/y for x,y in zip(no_right, no_total_model)])
print(no_right)
print(no_total)
print(no_total_model)

print(error_scenario)
#		f.write('%0.8f\t%0.8f\t%s\t%s\n'%(after_time-before_time,model_score,ground_truth,sentence))
f.close()
fe.close()
fip.close()
