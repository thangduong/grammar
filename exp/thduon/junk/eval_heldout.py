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
e.dump_variable_sizes()


params = { 'num_words_before': 5,
					 'num_words_after': 5,
					 'embedding_size': 300,
					 'vocab_size': 100000,
					 'embedding_device': None,
					 'batch_size': 128,
					 'num_classes': 2,
					 'mini_batches_between_checkpoint': 100,
					 'monolingual_dir': '/mnt/work/1-billion-word-language-modeling-benchmark'
					 }
test_data = ClassifierData.get_monolingual_test(params=params)
batch = test_data.next_batch(batch_size=50000)

model_results = []
f = open('heldout.txt', 'w')
for sentence, ground_truth in zip(batch['sentence'], batch['y']):
	_, indexed, _, _ = i.index_wordlist(sentence)
	before_time = time()
	r = e.eval({'sentence': [indexed]}, {'sm_decision'})
	after_time = time()
	model_score = r[0][0][1]
	model_results.append([indexed, model_score, ground_truth])
	f.write('%0.8f\t%0.8f\t%s\t%s\n'%(after_time-before_time,model_score,ground_truth,sentence))
f.close()

for thres in np.linspace(0,1,50):
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
	print('%s,%s,%s'%(thres,precision,recall))

