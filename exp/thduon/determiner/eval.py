from framework.evaluator import Evaluator
import framework.utils.common as utils
from framework.trainer import Trainer
import framework.subgraph.losses as losses
import logging
import sys
import os
import numpy as np
import math
from framework.utils.data.text_indexer import TextIndexer
import tensorflow as tf
e = Evaluator.load2("outputv0.ckpt")
i = TextIndexer.from_file('vocab.pkl')

sentence = "In simple terms , high precision means that an algorithm " \
			"returned substantially more relevant results than irrelevant ones , while" \
			" high recall means that an algorithm returned most of the relevant results ."

sentence = "In simple terms , high precision means that algorithm " \
			"returned substantially more relevant results than irrelevant ones , while" \
			" high recall means that algorithm returned most of relevant results ."

sentence = ['<pad>']*10 + sentence.split() + ['<pad>']*10
print(sentence)
for toki, tok in enumerate(sentence):
	if (toki > len(sentence)-20):
		break
	csentence = sentence[toki:(toki+20)]
	_,indexed,_,_ = i.index_wordlist(csentence)
	r = e.eval({'sentence': [indexed]}, {'sm_decision'})
	print("%s %s"%(np.argmax(r),csentence[10]))
	print(r)
