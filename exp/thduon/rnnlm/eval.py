from framework.utils.data.text_indexer import TextIndexer
from framework.evaluator import Evaluator
import framework.utils.common as utils
import os
import numpy as np
import time
import pickle
import math
run_server = False

params = utils.load_param_file('output/rnnlmV1/params.py')

vocab_file = os.path.join(utils.get_dict_value(params,'output_location'), 'vocab.pkl')
ckpt = os.path.join(utils.get_dict_value(params,'output_location'),
										utils.get_dict_value(params, 'model_name') + '.ckpt')


e = Evaluator.load2(ckpt)
i = TextIndexer.from_file(vocab_file)

num_before = utils.get_dict_value(params, "num_words_before")
num_after = utils.get_dict_value(params, "num_words_after")
pad_tok = utils.get_dict_value(params, "pad_tok", '<pad>')

#sentences = ['<s> The apple , which is rotten , is not edible .']
sentences = ['<s> I ate a pair of orange .'
							,'<s> I ate a pear of orange .'
							,'<s> I ate a pair of oranges .'
						 ]
cell_size = params['cell_size']
num_steps = params['num_steps']
num_layers = params['num_layers']
if os.path.exists('sent_list.pkl'):# and False:
	with open('sent_list.pkl','rb') as f:
		sentences = pickle.load(f)

all_sentence_prob = {}
for sentence in sentences:
	orig_sentence = sentence
	sentence = sentence.lower()
	state = np.zeros([num_layers, 2, 1, cell_size])
	tokens = sentence.split()
	sentence_len = len(tokens)
	sentence_prob = 1.0
	before = time.time()
	if (len(tokens)%num_steps) > 0:
		tokens += ['.'] * (num_steps-(len(tokens)%num_steps))
	_,indexed,num_unk,unk_words = i.index_wordlist(tokens)
	for toki in range((int(len(indexed)/num_steps))):
		r = e.eval({'x': [indexed[(toki*num_steps):(toki+1)*num_steps]],
								'state': state}, ['output_logits_sm', 'final_state'])
		state = r[1]
		for j in range(num_steps):
			#print(r[0][0][j][indexed[toki*20+j]])
			sentence_prob *= r[0][0][j][indexed[toki*20+j]]
			if toki*20+j > sentence_len:
				break
	diff = time.time() - before
	print("execute time = %s, unk_count = %s %s" % (diff, num_unk, unk_words))
	print("%s:%s:%s"%(sentence_prob, math.log(sentence_prob), sentence))
	all_sentence_prob[orig_sentence] = (sentence_prob)

with open('sent_prob.pkl', 'wb') as f:
	pickle.dump(all_sentence_prob,f)
