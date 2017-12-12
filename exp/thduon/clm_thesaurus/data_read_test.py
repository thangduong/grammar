from framework.utils.data.text_indexer import TextIndexer
from word_classifier.data import ClassifierData
import framework.subgraph.losses as losses
import framework.utils.common as utils
import data
from framework.trainer import Trainer, _default_train_iteration_done
from time import time
import pickle
import model
import os
import shutil
import copy
import numpy as np

param_file = 'params.py'
params = utils.load_param_file(param_file)
indexer = TextIndexer.from_txt_file(utils.get_dict_value(params, 'vocab_file'))
indexer.add_token('<pad>')
#indexer.add_token('unk')
#params['keywords'] = indexer.vocab_map()
keywords = []
with open('/mnt/work/1-billion-word-language-modeling-benchmark/lc_vocab_alpha.txt', 'r') as f:
	for line in f:
		line = line.rstrip().lstrip()
		pieces = line.split()
		word = pieces[0]
		if word.isalpha():
			keywords.append(word)
params['keywords'] = keywords
params['num_classes'] = len(params['keywords'])+1
os.makedirs(utils.get_dict_value(params,'output_location'), exist_ok=True)
indexer.save_vocab_as_pkl(os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl'))
params['vocab_size'] = indexer.vocab_size()
training_data = ClassifierData.get_monolingual_training(base_dir=params['monolingual_dir'],
																												indexer=indexer,
																												params=params,
																												gen_data_fcn=data.gen_data)
before = time()
b = training_data.next_batch(1024)
after = time()
print(after-before)