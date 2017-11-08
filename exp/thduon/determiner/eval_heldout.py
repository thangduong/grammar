from framework.evaluator import Evaluator
from classifier_data import ClassifierData
import framework.utils.common as utils
from framework.trainer import Trainer
import framework.subgraph.losses as losses
import logging
import sys
import os
from framework.utils.data.text_indexer import TextIndexer
import tensorflow as tf
import numpy as np
from time import time

e = Evaluator.load2("outputv0.ckpt")
i = TextIndexer.from_file('vocab.pkl')
#e.dump_variable_sizes()


params = { 'num_words_before': 10,
					 'num_words_after': 10,
					 'embedding_size': 300,
					 'vocab_size': 100000,
					 'embedding_device': None,
					 'batch_size': 128,
					 'num_classes': 2,
					 'keywords': ['a', 'an', 'the'],
					 'mini_batches_between_checkpoint': 100,
					 'monolingual_dir': '/mnt/work/1-billion-word-language-modeling-benchmark'
					 }
test_data = ClassifierData.get_monolingual_test(params=params)
alist = []
for cur_batch in range(10):
	batch = test_data.next_batch(batch_size=5000)

	model_results = []
	f = open('heldout.txt', 'w')
	mf = open('heldout_mistakes.txt', 'w')
	correct = 0
	incorrect = 0
	lcorrect = [0,0,0,0]
	lincorrect = [0,0,0,0]
	for sentence, ground_truth in zip(batch['sentence'], batch['y']):
		_, indexed, _, _ = i.index_wordlist(sentence)
		before_time = time()
		r = e.eval({'sentence': [indexed]}, {'sm_decision'})
		after_time = time()
		model_sm = r[0][0]
		model_score = np.argmax(model_sm)
		model_results.append([indexed, model_score, ground_truth])
		sentence[10] = '***'+sentence[10]+"***"
		s = ('%0.8f\t%s\t%s\t%s'%(after_time-before_time,model_score,ground_truth,sentence))
		if model_score == ground_truth:
			correct += 1
			lcorrect[ground_truth] += 1
		else:
			incorrect += 1
			lincorrect[ground_truth] += 1
			mf.write('%s\n'%s)
			mf.write('%s\n'%model_sm)
		f.write('%s\n'%s)
	#	print(s)
	f.close()
	mf.close()
	print('%s %s %s'%(correct, incorrect, correct/(correct+incorrect)))
	alist.append(correct/(correct+incorrect))
	print(lcorrect)
	print(lincorrect)

print(alist)
print(np.mean(alist))
print(np.std(alist))