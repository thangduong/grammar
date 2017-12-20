from framework.utils.data.text_indexer import TextIndexer
from framework.evaluator import Evaluator
import framework.utils.common as utils
import os
import numpy as np
run_server = False

params = utils.load_param_file('output/rnnlmV0/params.py')

vocab_file = os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl')
ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')


e = Evaluator.load2(ckpt)
i = TextIndexer.from_file(vocab_file)

num_before = utils.get_dict_value(params, "num_words_before")
num_after = utils.get_dict_value(params, "num_words_after")
pad_tok = utils.get_dict_value(params, "pad_tok", '<pad>')

sentences = ['<s> The apple , which is rotten , is not edible']
for sentence in sentences:
	_,indexed,_,_ = i.index_wordlist(csentence)
	r = e.eval({'x': [indexed]}, {'output_logits'})
	rr = r[0][0]
