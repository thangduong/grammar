from framework.evaluator import Evaluator
import framework.utils.common as utils
from framework.trainer import Trainer
import framework.subgraph.losses as losses
import logging
import sys
import os
from framework.utils.data.text_indexer import TextIndexer
import tensorflow as tf
sentences = ['Jimenez reported from San Jose Costa Rica ; Associated Press',
             "We went to the store and I bought a banana",
						 "This is the cat that walked over the mat .",
						 "<pad> <pad> In this case I think the problem is"]
e = Evaluator.load2("outputv0.ckpt")
i = TextIndexer.from_file('vocab.pkl')

for sentence in sentences:
#	for j in range(5):
#		sentence = '<pad> ' + sentence + " <pad>"
	_,indexed,_,_ = i.index_wordlist(sentence.split())
	r = e.eval({'sentence': [indexed]}, {'sm_decision'})
	print(sentence)
	print(r)