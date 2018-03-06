from framework.utils.data.text_indexer import TextIndexer
from word_classifier.data import ClassifierData
from framework.evaluator import Evaluator
from shell_command import shell_call
import framework.utils.common as utils
from time import time
import numpy as np
import os
import sys

params = utils.load_param_file(sys.argv[1])

params['num_classes'] = len(params['keywords'])+1
vocab_file = os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl')
release_dir = os.path.join(utils.get_dict_value(params,'output_location'), params['model_name'])
graphdef_file = os.path.join(release_dir, params['model_name'] + '.graphdef')
ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')

e = Evaluator.load_graphdef(graphdef_file)
e.dump_variable_sizes()
i = TextIndexer.from_file(vocab_file)

#test_data = ClassifierData.get_data_from_dirs(['/mnt/work/training_data/enron.tokenized/valid'],params=params)
#test_data = ClassifierData.get_data_from_dirs(['/mnt/work/training_data/enron.test.tokenized'],params=params)
test_data = ClassifierData(file_list=['/mnt/work/training_data/oxoml-enron-sentsentences.test.v2.tokenized.txt'], params=params)
model_results = []

timestr = str(int(time()))
num_classes = params['num_classes']
no_right = [0]*num_classes
no_total = [0]*num_classes
no_total_model = [0]*num_classes


error_scenario = []
for x in range(num_classes):
	error_scenario += [[0] * num_classes]

topn = 1
last = 0
total_indexed = 0
total_unk = 0
all_unk_words = []

right = total = 0
fefn = '/tmp/eval_heldout3.%s.txt'%time()
print('writing to %s'%fefn)
fe = open(fefn,'w')
for batch_no in range(1):
	print("WORKING ON BATCH %s" % batch_no)
	batch = test_data.next_batch(batch_size=200000)
	for idx, (sentence, ground_truth, word) in enumerate(zip(batch['sentence'], batch['y'], batch['word'])):
		num_indexed, indexed, num_unk, unk_words = i.index_wordlist(sentence)
		total_indexed += num_indexed
		total_unk += num_unk
		all_unk_words += unk_words
		before_time = time()
		r = e.eval({'sentence': [indexed], 'word': [word]}, {'sm_decision'})
		after_time = time()
		dt = after_time-before_time
		pick = np.argpartition(r[0][0], -topn)[-topn:]
#		pick = np.argmax(r[0][0])
		pickpr = [r[0][0][idx] for idx in pick]
		model_score = r[0][0]
		x = [dt, pick, pickpr, ground_truth, num_unk, num_indexed,
				 ' '.join(sentence[:int(len(sentence)/2)]+['___']+sentence[int(len(sentence)/2):]),\
				 r[0][0]]
		model_pick = pick[0]
		total += 1
		if ground_truth in pick:
			model_pick = ground_truth
			no_right[ground_truth] += 1
			right += 1
		else:
			fe.write('%s\n' % x)
			fe.flush()
		if (ground_truth == 1 and pick[0] == 2) or (ground_truth == 2 and pick[0] == 1):
			print([idx, ground_truth, pick, pickpr, r[0][0]])
			print([chr(x) for x in word])
			print(sentence)
		error_scenario[ground_truth][pick[0]] += 1
		no_total[ground_truth] += 1
		no_total_model[model_pick] += 1
		model_results.append(x)
		if idx-last > 10000:
#			print(all_unk_words)
			print("NUMBER OF UNK: %d (%0.2f)" % (total_unk, 100 * total_unk / max(total_indexed, 1)))
			print("recall = %s" % [x / max(y,1) for x, y in zip(no_right, no_total)])
			print("precision = %s" % [x / max(y,1) for x, y in zip(no_right, no_total_model)])
			print(no_right)
			print(no_total)
			print(no_total_model)
			print(error_scenario)
			print("ACCURACY = %s" % (right/total))
			last =idx
fe.close()
#for x in model_results:
#	print(x)
print("recall = %s"%[x/max(y,1) for x,y in zip(no_right,no_total)])
print("precision = %s"%[x/max(y,1) for x,y in zip(no_right, no_total_model)])
print(no_right)
print(no_total)
print(no_total_model)

print(error_scenario)
print("ACCURACY = %s" % (right / total))

# timestamp, precision, recall, precision, recall, precision, recall, precision, recall
precision_list = [x/max(y,1) for x,y in zip(no_right,no_total)]
recall_list = [x/max(y,1) for x,y in zip(no_right, no_total_model)]

s = str(time())
for x in [x for pair in zip(precision_list, recall_list) for x in pair]:
	s += ' ' + str(x)
s += ' '+str(100.0*right/total)
print(s)
